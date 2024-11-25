class TestGenerator:
    def __init__(self):
        # mimic data
        self.all_test_cases = [
            "test_case_1", "test_case_2", "test_case_3", "test_case_4", "test_case_5"
        ]
        self.all_test_score = [
            4.5, 2.3, 3.8, 1.2, 5.0
        ]
        self.output_num = 3  # select 3 

    def get_most_challenging_cases(self):
        # final sort all test_cases and return most challenging case
        # first zip
        test_with_score = list(zip(self.all_test_cases, self.all_test_score))
        # test_with_score = list(zip(self.all_test_cases, self.all_test_dist)) # Here we can also use test min_dist to sort
        # sort
        sorted_test_with_score = sorted(test_with_score, key=lambda x:x[1])
        # unzip
        sorted_test_cases = [x[0] for x in sorted_test_with_score]
        sorted_score = [x[1] for x in sorted_test_with_score]
        # pick top cases
        top_n_test_cases = sorted_test_cases[:self.output_num]
        
        # print selected test cases
        print("Sorted Test Cases:", sorted_test_cases)
        print("Sorted Scores:", sorted_score)
        print("Top N Test Cases:", top_n_test_cases)
        
        return top_n_test_cases


test_generator = TestGenerator()
top_cases = test_generator.get_most_challenging_cases()
