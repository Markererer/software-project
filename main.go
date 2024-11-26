package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func main() {
	// Initialize Dagger client
	ctx := context.Background()
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(nil))
	if err != nil {
		log.Fatalf("Failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// Dummy pipeline: Run "echo Hello World!" in an Alpine container
	output, err := client.Container().
		From("alpine").
		WithExec([]string{"echo", "Hello World! John Dagger here."}).
		Stdout(ctx)
	if err != nil {
		log.Fatalf("Failed to run pipeline: %v", err)
	}

	// Print pipeline output
	log.Println("Pipeline output:", output)
}
