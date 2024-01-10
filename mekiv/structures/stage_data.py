from netaddr import Z
import torch

from dataclasses import dataclass, fields, Field
from typing import Optional, Callable, Any, Tuple, Iterable
from typing_extensions import Self

from mekiv.utils.misc import rand_split


def _tensor_fields(obj: Any) -> Tuple[Field[torch.Tensor]]:
    """Returns all tensor fields of an object."""
    return tuple(
        field
        for field in fields(obj)
        if isinstance(getattr(obj, field.name), torch.Tensor)
    )


class _TensorDataclass:
    """
    Base class for dataclasses with tensor fields.
    Provides convenience methods for:
        - Moving all tensor fields to a device
        - Converting all tensor fields to float on initialisation
        - Minibatches
        - Ensuring that all tensor fields have the same length
    """

    def __post_init__(self):
        self._length = None

        for field in _tensor_fields(self):
            field_attr = getattr(self, field.name)
            setattr(self, field.name, field_attr.float())

            if self._length is None:
                self._length = len(field_attr)
            else:
                assert len(field_attr) == self._length, (
                    f"Field '{field.name}' has length {len(field_attr)}, "
                    f"but expected {self._length}."
                )

    def to(self, device: str | torch.device) -> None:
        """Moves all tensor fields to a device."""
        for field in _tensor_fields(self):
            setattr(self, field.name, getattr(self, field.name).to(device))

    def __getitem__(self, indices: int | slice | Iterable[int]) -> Self:
        """
        Returns a new instance of the class with all tensor fields indexed.
        """
        tensors = {
            field.name: getattr(self, field.name)[indices]
            for field in _tensor_fields(self)
        }
        others = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in tensors
        }
        return type(self)(**tensors, **others)

    def __len__(self) -> int:
        return self._length

    def minibatches(self, batch_size: int, shuffle: bool = True) -> Iterable[Self]:
        """Returns an iterable of minibatches of data."""
        indices = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for minibatch in indices.split(batch_size):
            yield self[minibatch]


@dataclass
class KIVStage1Data(_TensorDataclass):
    X: torch.Tensor
    Z: torch.Tensor
    Y: Optional[torch.Tensor] = None


@dataclass
class KIVStage2Data(_TensorDataclass):
    Y: torch.Tensor
    Z: torch.Tensor
    X: Optional[torch.Tensor] = None


@dataclass
class KIVStageData:
    stage_1: KIVStage1Data
    stage_2: KIVStage2Data

    @property
    def all_X(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.X, self.stage_2.X))

    @property
    def all_Y(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.Y, self.stage_2.Y))

    @property
    def all_Z(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.Z, self.stage_2.Z))

    def to(self, device: str | torch.device) -> None:
        self.stage_1.to(device)
        self.stage_2.to(device)

    @classmethod
    def from_all_data(
        cls,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        seed: int = 42,
        p: float = 0.5,
    ) -> Self:
        assert len(X) == len(Y) == len(Z)
        assert 0 <= p <= 1

        first, second = rand_split((X, Y, Z), p=p, seed=seed)
        X1, Y1, Z1 = first
        X2, Y2, Z2 = second

        return cls(KIVStage1Data(X=X1, Y=Y1, Z=Z1), KIVStage2Data(X=X2, Y=Y2, Z=Z2))


@dataclass
class TestData(_TensorDataclass):
    X: torch.Tensor
    truth: torch.Tensor
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def evaluate_preds(self, preds: torch.Tensor) -> float:
        """Evaluates a set of predictions against the truth."""
        preds = preds.to(self.truth.device)
        return self.metric(preds, self.truth).to("cpu").item()
