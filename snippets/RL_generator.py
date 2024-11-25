import random
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RLGenerator(object):
    min_size = Obstacle.Size(2, 2, 10)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(-40, 10, 0, 0)
    max_position = Obstacle.Position(30, 40, 0, 90)

    def __init__(self, case_study_file: str) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)

    def generate(self, budget: int) -> List[TestCase]:
        test_cases = []
        for i in range(budget):
            size = Obstacle.Size(
                l=random.uniform(self.min_size.l, self.max_size.l),
                w=random.uniform(self.min_size.w, self.max_size.w),
                h=random.uniform(self.min_size.h, self.max_size.h),
            )
            position = Obstacle.Position(
                x=random.uniform(self.min_position.x, self.max_position.x),
                y=random.uniform(self.min_position.y, self.max_position.y),
                z=0,  # obstacles should always be place on the ground
                r=random.uniform(self.min_position.r, self.max_position.r),
            )
            obstacle = Obstacle(size, position)
            test = TestCase(self.case_study, [obstacle])
            try:
                test.execute()
                distances = test.get_distances()
                print(f"minimum_distance:{min(distances)}")
                test.plot()
                test_cases.append(test)
            except Exception as e:
                print("Exception during test execution, skipping the test")
                print(e)

        ### You should only return the test cases
        ### that are needed for evaluation (failing or challenging ones)
        return test_cases


if __name__ == "__main__":
    generator = RandomGenerator("case_studies/mission1.yaml")
    generator.generate(3)