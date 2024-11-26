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

	// Placeholder for actual stuff we add later
	log.Println("Hello world!")
}
