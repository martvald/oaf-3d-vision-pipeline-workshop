from __future__ import annotations

from dataclasses import dataclass

from oaf_vision_3d.tests.status import Status
from oaf_vision_3d.tests.test_workshop_02 import workshop_02_results
from oaf_vision_3d.tests.test_workshop_03 import workshop_03_results
from oaf_vision_3d.tests.test_workshop_04 import workshop_04_results
from oaf_vision_3d.tests.test_workshop_05 import workshop_05_results
from oaf_vision_3d.tests.test_workshop_07 import workshop_07_results


@dataclass
class WorkshopResult:
    workshop_01: Status = Status.PASSED
    workshop_02: Status = workshop_02_results()
    workshop_03: Status = workshop_03_results()
    workshop_04: Status = workshop_04_results()
    workshop_05: Status = workshop_05_results()
    workshop_06: Status = Status.NOT_STARTED
    workshop_07: Status = workshop_07_results()
    workshop_08: Status = Status.NOT_STARTED

    def __post_init__(self) -> None:
        if (
            self.workshop_02 is Status.PASSED
            and self.workshop_03 is Status.PASSED
            and self.workshop_04 is Status.PASSED
            and self.workshop_05 is Status.PASSED
        ):
            self.workshop_06 = Status.PASSED
        if self.workshop_06 is Status.PASSED and self.workshop_07 is Status.PASSED:
            self.workshop_08 = Status.PASSED

    def __str__(self) -> str:
        return (
            f"Workshop 01: {self.workshop_01.name}\n"
            f"Workshop 02: {self.workshop_02.name}\n"
            f"Workshop 03: {self.workshop_03.name}\n"
            f"Workshop 04: {self.workshop_04.name}\n"
            f"Workshop 05: {self.workshop_05.name}\n"
            f"Workshop 06: {self.workshop_06.name}\n"
            f"Workshop 07: {self.workshop_07.name}\n"
            f"Workshop 08: {self.workshop_08.name}"
        )


if __name__ == "__main__":
    print(WorkshopResult())
