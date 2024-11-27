package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

const (
	// Preprocessing
	rawDataDir       = "./data/raw"                 // Local path to raw data
	artifactsDir     = "./data/artifacts"           // Local path to save artifacts
	processedDataDir = "./data/processed"           // Local path to save processed data
	scriptPath       = "./scripts/preprocessing.py" // Path to the Python script
	minDate          = "2024-01-01"                 // Example date range
	maxDate          = "2024-01-31"
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

	// Run the Python script with arguments
	output, err := pythonContainer.
		WithExec([]string{
			"python", scriptPath,
			"--raw_data_dir", rawDataDir,
			"--artifacts_dir", artifactsDir,
			"--processed_data_dir", processedDataDir,
			"--min_date", minDate,
			"--max_date", maxDate,
		}).
		Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run preprocessing script: %v", err)
	}

	// Print pipeline output
	log.Println("Pipeline output:", output)
}
