from dataclasses import dataclass

from Bloom.BaseBloomFilter import BloomBase, MembershipFilter
from Bloom.Exceptions import IncompatibleFilterError, ParameterTypeError
from Bloom.Hashing import MAX_SEED
from Bloom.Standard import BloomFilter
from Bloom.Theory import combined_false_positive_rate, split_error_budget
from Bloom.Validation import require_int


@dataclass(frozen=True)
class CrossCheckResult:
    """Verdicts of the two member filters for a single item.

    Attributes:
        primary: Verdict of the primary filter.
        verifier: Verdict of the verifier filter.
    """

    primary: bool
    verifier: bool

    @property
    def present(self) -> bool:
        """True only when both filters accept the item."""
        return self.primary and self.verifier

    @property
    def agreed(self) -> bool:
        """True when both filters gave the same verdict."""
        return self.primary == self.verifier


class CrossCheckedBloomFilter(MembershipFilter):
    """Two independently seeded filters that verify each other.

    Every item is inserted into both filters. A query is accepted only when
    both agree it is present. Because an inserted item is always accepted by
    both, any disagreement proves the item was never inserted and exposes
    which filter produced a false positive. The check method records these
    catches so the real false positive behaviour can be observed.

    The combined false positive rate is the product of the two member rates,
    valid because the seeds differ and the hash families are independent.

    Args:
        primary: The first member filter.
        verifier: The second member filter, with a different seed.

    Raises:
        ParameterTypeError: If a member is not a BloomBase filter.
        IncompatibleFilterError: If both members share a seed.
    """

    def __init__(self, primary: BloomBase, verifier: BloomBase) -> None:
        for name, member in (("primary", primary), ("verifier", verifier)):
            if not isinstance(member, BloomBase):
                raise ParameterTypeError(
                    f"{name} must be a BloomBase filter, got {type(member).__name__}",
                    parameter=name,
                    value=member,
                )

        if primary.seed == verifier.seed:
            raise IncompatibleFilterError(
                "primary and verifier must use different seeds to be independent, "
                f"both use {primary.seed}"
            )

        self._primary = primary
        self._verifier = verifier
        self._checks = 0
        self._disagreements = 0

    @classmethod
    def from_capacity(
        cls,
        capacity: int,
        error_rate: float,
        seed: int = 0,
        primary_type: type[BloomBase] = BloomFilter,
        verifier_type: type[BloomBase] = BloomFilter,
    ) -> "CrossCheckedBloomFilter":
        """Build a pair of filters whose combined rate meets error_rate.

        The error budget is split equally, so each member targets the square
        root of error_rate. The members use seeds seed and seed plus one.

        Args:
            capacity: Number of distinct items to hold.
            error_rate: Combined false positive target, strictly between
                0 and 1.
            seed: Seed of the primary filter. The verifier uses seed plus one.
            primary_type: Filter class used for the primary member.
            verifier_type: Filter class used for the verifier member.

        Raises:
            InvalidParameterError: If an argument is invalid.
        """
        require_int("seed", seed, minimum=0, maximum=MAX_SEED - 1)
        member_rate = split_error_budget(error_rate, 2)

        return cls(
            primary_type.from_capacity(capacity, member_rate, seed=seed),
            verifier_type.from_capacity(capacity, member_rate, seed=seed + 1),
        )

    @property
    def primary(self) -> BloomBase:
        """The primary member filter."""
        return self._primary

    @property
    def verifier(self) -> BloomBase:
        """The verifier member filter."""
        return self._verifier

    @property
    def item_count(self) -> int:
        return self._primary.item_count

    @property
    def memory_bits(self) -> int:
        return self._primary.memory_bits + self._verifier.memory_bits

    @property
    def memory_bytes(self) -> int:
        return self._primary.memory_bytes + self._verifier.memory_bytes

    @property
    def checks(self) -> int:
        """Number of calls to check since creation or the last clear."""
        return self._checks

    @property
    def caught_false_positives(self) -> int:
        """Number of checks where the members disagreed."""
        return self._disagreements

    @property
    def disagreement_rate(self) -> float:
        """Fraction of checks where the members disagreed."""
        return self._disagreements / self._checks if self._checks else 0.0

    def add(self, item: object) -> bool:
        primary_changed = self._primary.add(item)
        verifier_changed = self._verifier.add(item)

        return primary_changed or verifier_changed

    def check(self, item: object) -> CrossCheckResult:
        """Query both filters and record whether they agreed.

        Unlike the in operator, this method updates the disagreement
        statistics.

        Raises:
            InvalidItemError: If the item type is unsupported.
        """
        result = CrossCheckResult(item in self._primary, item in self._verifier)

        self._checks += 1

        if not result.agreed:
            self._disagreements += 1

        return result

    def __contains__(self, item: object) -> bool:
        return item in self._primary and item in self._verifier

    def clear(self) -> None:
        self._primary.clear()
        self._verifier.clear()
        self._checks = 0
        self._disagreements = 0

    def expected_false_positive_rate(self) -> float:
        return combined_false_positive_rate(
            (
                self._primary.expected_false_positive_rate(),
                self._verifier.expected_false_positive_rate(),
            )
        )

    def __repr__(self) -> str:
        return (
            f"CrossCheckedBloomFilter(primary={self._primary!r}, "
            f"verifier={self._verifier!r})"
        )
