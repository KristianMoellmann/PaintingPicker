# Define the arguments
$names = @("kristoffer", "Kristian", "kasper")
$score_types = @("original", "logic", "clip")
$regressors = @("ScaleNetV2")

# Loop through each argument
foreach ($name in $names) {
    foreach ($score_type in $score_types) {
        foreach ($regressor in $regressors) {
            # Run the command with the current arguments
            python src/train_model.py $name --scoring elo --score_type $score_type --regressor $regressor --dont_plot --epochs 200 --save_results
        }
    }
}

# Loop through each argument
foreach ($name in $names) {
    foreach ($regressor in $regressors) {
        # Run the command with the current arguments
        python src/train_model.py $name --scoring scale_9 --regressor $regressor --dont_plot --epochs 200 --save_results
    }
}