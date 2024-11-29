package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

const (
	rawDataDir       = "./data/raw"       // Local path to raw data
	artifactsDir     = "./data/artifacts" // Local path to save artifacts
	interimDir       = "./data/interim"   // Local path to save interim data
	processedDataDir = "./data/processed" // Local path to save processed data

	// Preprocessing
	preprocessingScriptPath = "./scripts/preprocessing.py" // Path to the data preprocessing script
	minDate                 = "2024-01-01"                 // Date range
	maxDate                 = "2024-01-31"

	// Features
	featuresScriptPath = "./scripts/features.py" // Path to the features extraction script
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

	output, err := pythonContainer.
		WithExec([]string{
			"python", "-u", preprocessingScriptPath,
			"--raw_data_dir", rawDataDir,
			"--artifacts_dir", artifactsDir,
			"--processed_data_dir", processedDataDir,
			"--min_date", minDate,
			"--max_date", maxDate,
		}).
		WithExec([]string{
			"python", "-u", featuresScriptPath,
			"--raw_data_dir", rawDataDir,
			"--interim_data_dir", interimDir,
			"--processed_data_dir", processedDataDir,
		}).
		Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run scripts: %v", err)
	}

	// Print pipeline output
	log.Println("Pipeline output:", output)
}
