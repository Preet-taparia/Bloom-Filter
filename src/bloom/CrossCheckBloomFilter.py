from collections.abc import Iterable, Sequence

from Bloom.BaseBloomFilter import BloomBase
from Bloom.Exceptions import (
    IncompatibleFilterError,
    ParameterTypeError,
    ParameterValueError,
)
from Bloom.Hashing import MAX_SEED
from Bloom.Standard import BloomFilter
from Bloom.Theory import combined_false_positive_rate, split_error_budget
from Bloom.Validation import require_int


class CrossCheckedBloomFilter:
    """Several independently seeded filters that verify each other.

    Every item is inserted into all members. A query is accepted only when
    all of them agree it is present. Because an inserted item is always
    accepted by every member, any disagreement proves the item was never
    inserted and exposes which members produced a false positive. The check
    method records these catches so the real false positive behaviour can be
    observed.

    The combined false positive rate is the product of the member rates,
    valid because the seeds differ and the hash families are independent.

    Args:
        members: Two or more member filters, each with a different seed.

    Raises:
        ParameterTypeError: If a member is not a BloomBase filter.
        ParameterValueError: If fewer than two members are given.
        IncompatibleFilterError: If any two members share a seed.
    """

    def __init__(self, members: Sequence[BloomBase]) -> None:
        members = tuple(members)

        if len(members) < 2:
            raise ParameterValueError(
                f"at least 2 members are required, got {len(members)}",
                parameter="members",
                value=members,
            )

        for index, member in enumerate(members):
            if not isinstance(member, BloomBase):
                raise ParameterTypeError(
                    f"member {index} must be a BloomBase filter, "
                    f"got {type(member).__name__}",
                    parameter="members",
                    value=member,
                )

        seeds = [member.seed for member in members]

        if len(set(seeds)) != len(seeds):
            raise IncompatibleFilterError(
                f"members must use different seeds to be independent, got {seeds}"
            )

        self._members = members
        self._checks = 0
        self._disagreements = 0
        self._member_false_positives = [0] * len(members)

    @classmethod
    def from_capacity(
        cls,
        capacity: int,
        error_rate: float,
        seed: int = 0,
        filter_count: int = 2,
        member_types: Sequence[type[BloomBase]] | None = None,
    ) -> "CrossCheckedBloomFilter":
        """Build filters whose combined rate meets error_rate.

        The error budget is split equally, so each member targets the
        filter_count th root of error_rate. Member i uses seed plus i.

        Args:
            capacity: Number of distinct items to hold.
            error_rate: Combined false positive target, strictly between
                0 and 1.
            seed: Seed of the first member.
            filter_count: Number of members, at least 2.
            member_types: Filter class of each member. Defaults to
                BloomFilter for all of them.

        Raises:
            InvalidParameterError: If an argument is invalid.
        """
        filter_count = require_int("filter_count", filter_count, minimum=2)
        require_int("seed", seed, minimum=0, maximum=MAX_SEED - filter_count + 1)

        if member_types is None:
            member_types = (BloomFilter,) * filter_count
        elif len(member_types) != filter_count:
            raise ParameterValueError(
                f"member_types has {len(member_types)} entries "
                f"but filter_count is {filter_count}",
                parameter="member_types",
                value=member_types,
            )

        member_rate = split_error_budget(error_rate, filter_count)

        return cls(
            [
                member_type.from_capacity(capacity, member_rate, seed=seed + offset)
                for offset, member_type in enumerate(member_types)
            ]
        )

    @property
    def members(self) -> tuple[BloomBase, ...]:
        """The member filters, in order."""
        return self._members

    @property
    def member_count(self) -> int:
        """Number of member filters."""
        return len(self._members)

    @property
    def item_count(self) -> int:
        """Number of insertions performed since creation or the last clear."""
        return self._members[0].item_count

    @property
    def memory_bits(self) -> int:
        """Theoretical storage requirement of all members in bits."""
        return sum(member.memory_bits for member in self._members)

    @property
    def memory_bytes(self) -> int:
        """Theoretical storage requirement of all members in bytes."""
        return sum(member.memory_bytes for member in self._members)

    @property
    def checks(self) -> int:
        """Number of calls to check since creation or the last clear."""
        return self._checks

    @property
    def caught_false_positives(self) -> int:
        """Number of checks where the members disagreed."""
        return self._disagreements

    @property
    def member_false_positives(self) -> tuple[int, ...]:
        """Per member count of false positives exposed by disagreements."""
        return tuple(self._member_false_positives)

    @property
    def disagreement_rate(self) -> float:
        """Fraction of checks where the members disagreed."""
        return self._disagreements / self._checks if self._checks else 0.0

    def add(self, item: object) -> bool:
        """Insert an item into every member.

        Returns:
            True if any member changed.
        """
        changed = False

        for member in self._members:
            if member.add(item):
                changed = True

        return changed

    def update(self, items: Iterable[object]) -> None:
        """Insert every item from an iterable."""
        for item in items:
            self.add(item)

    def check(self, item: object) -> tuple[bool, ...]:
        """Query every member and record whether they agreed.

        Unlike the in operator, this method updates the disagreement
        statistics. When the verdicts differ, the item was never inserted
        and every member that accepted it is counted as a false positive.

        Returns:
            The verdict of each member, in member order.

        Raises:
            InvalidItemError: If the item type is unsupported.
        """
        verdicts = tuple(item in member for member in self._members)

        self._checks += 1

        if len(set(verdicts)) > 1:
            self._disagreements += 1

            for index, verdict in enumerate(verdicts):
                if verdict:
                    self._member_false_positives[index] += 1

        return verdicts

    def clear(self) -> None:
        """Remove every item and reset all statistics."""
        for member in self._members:
            member.clear()

        self._checks = 0
        self._disagreements = 0
        self._member_false_positives = [0] * len(self._members)

    def expected_false_positive_rate(self) -> float:
        """Theoretical false positive probability at the current load."""
        return combined_false_positive_rate(
            tuple(member.expected_false_positive_rate() for member in self._members)
        )

    def __contains__(self, item: object) -> bool:
        return all(item in member for member in self._members)

    def __len__(self) -> int:
        return self.item_count

    def __repr__(self) -> str:
        members = ", ".join(repr(member) for member in self._members)

        return f"CrossCheckedBloomFilter(members=[{members}])"
