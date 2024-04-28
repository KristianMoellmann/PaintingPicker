import os
import yaml
import numpy as np


def evaluate_results(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
        for sim_type, sym_data in data.items():
            print(f"    Simulation type: {sim_type}")
            for model, model_data in sym_data.items():
                print(f"        Model: {model}")
                model_avg_test_loss = []
                # for person, person_data in model_data.items():
                for person in ['kasper', 'Kristian', 'kristoffer']:
                    print(f"            Person: {person}")
                    person_data = model_data[person]
                    test_losses = person_data['test_losses']
                    avg_test_loss = sum(test_losses) / len(test_losses)
                    std_test_loss = np.std(test_losses)
                    print(f"                Test - Avg: {avg_test_loss:.3f}, Std: {std_test_loss:.3f}")
                    model_avg_test_loss.append(avg_test_loss)

                    test_kendalls_tau_corr = person_data['test_kendalls_tau_corr']
                    avg_kendalls_tau_corr = sum(test_kendalls_tau_corr) / len(test_kendalls_tau_corr)
                    std_kendalls_tau_corr = np.std(test_kendalls_tau_corr)
                    print(f"                Corr - Avg: {avg_kendalls_tau_corr:.2f}, Std: {std_kendalls_tau_corr:.2f}")
                print(f"            Model - Avg: {sum(model_avg_test_loss) / len(model_avg_test_loss):.3f}")
                print("")
            print("")

if __name__ == "__main__":
    results_dir = "results/full/"
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".yaml"):
            print(f"Evaluating results in file: {file_name}")
            file_path = os.path.join(results_dir, file_name)
            evaluate_results(file_path)