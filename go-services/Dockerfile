# Stage 1: Build the Go application
FROM golang:1.17-alpine AS go-builder

WORKDIR /app

# Install Go dependencies
COPY go.mod go.sum ./
RUN go mod tidy

# Copy and build the application
COPY . ./
RUN go build -o /main

# Final Stage: Create a minimal image for the Go service
FROM alpine:latest

# Copy the Go binary from the builder stage
COPY --from=go-builder /main /main

# Set the working directory
WORKDIR /app

# Expose the necessary port for the Go service
EXPOSE 8080

# Command to run the Go service
CMD ["/main"]
