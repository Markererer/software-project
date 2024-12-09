package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

const (
	// Folder paths
	rawDataDir       = "./data/raw"       // Path to raw data
	artifactsDir     = "./artifacts"      // Path to save artifacts
	interimDir       = "./data/interim"   // Path to save interim data
	processedDataDir = "./data/processed" // Path to save processed data
	modelsDir        = "./models"         // Path to save models
	mlrunsDir        = "./mlruns"         // Path to save MLFlow runs

	// Python scripts
	preprocessingScriptPath = "./scripts/preprocessing.py" // Path to the data preprocessing script
	featuresScriptPath      = "./scripts/features.py"      // Path to the features extraction script
	trainingScriptPath      = "./scripts/train.py"         // Path to the model training script
	evaluationScriptPath    = "./scripts/evaluation.py"    // Path to the model evaluation script
	deploymentScriptPath    = "./scripts/deployment.py"    // Path to the model deployment script

	// Script parameters
	minDate = "2024-01-01" // Date range for the preprocessing script
	maxDate = "2024-01-31"
)

func main() {
	// Initialize Dagger client
	ctx := context.Background()
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(nil))
	if err != nil {
		log.Fatalf("Failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// Define Python container
	pythonContainer := client.Container().
		From("python:3.12-slim"). // Use Python image
		WithMountedDirectory("/app", client.Host().Directory(".")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// Run the preprocessing script and get the updated container
	preprocessingContainer := pythonContainer.WithExec([]string{
		"python", "-u", preprocessingScriptPath,
		"--raw_data_dir", rawDataDir,
		"--artifacts_dir", artifactsDir,
		"--processed_data_dir", processedDataDir,
		"--min_date", minDate,
		"--max_date", maxDate,
	})

	// Get the output from the preprocessing script
	output, err := preprocessingContainer.Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run preprocessing script: %v", err)
	}
	log.Println("Preprocessing output:", output)

	// Run the feature extraction script on the updated container
	featuresContainer := preprocessingContainer.WithExec([]string{
		"python", "-u", featuresScriptPath,
		"--raw_data_dir", rawDataDir,
		"--interim_data_dir", interimDir,
		"--processed_data_dir", processedDataDir,
	})

	// Get the output from the feature extraction script
	output, err = featuresContainer.Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run features script: %v", err)
	}
	log.Println("Feature extraction output:", output)

	// Run the model training script on the updated container
	modelsContainer := featuresContainer.WithExec([]string{
		"python", "-u", trainingScriptPath,
		"--raw_data_dir", rawDataDir,
		"--interim_data_dir", interimDir,
		"--processed_data_dir", processedDataDir,
		"--artifacts_dir", artifactsDir,
		"--mlruns_dir", mlrunsDir,
		"--models_dir", modelsDir,
	})

	// Get the output from the model training script
	output, err = modelsContainer.Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run model training script: %v", err)
	}
	log.Println("Model training output:", output)

	// Run the model evaluation script on the updated container
	evalContainer := modelsContainer.WithExec([]string{
		"python", "-u", evaluationScriptPath,
		"--artifacts_dir", artifactsDir,
	})

	// Get the output from the model evaluation script
	output, err = evalContainer.Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run model training script: %v", err)
	}
	log.Println("Model evaluation output:", output)

	// Run the model deployment script on the updated container
	deployContainer := evalContainer.WithExec([]string{
		"python", "-u", deploymentScriptPath,
	})

	// Get the output from the model deployment script
	output, err = deployContainer.Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run model training script: %v", err)
	}
	log.Println("Model deployment output:", output)
}
