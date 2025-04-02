# Step 1: Build the application
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Step 2: Install 'serve' to serve static files
FROM node:22-alpine

# Install serve globally
RUN npm install -g serve

# Set the working directory and copy the build files from the previous stage
WORKDIR /app
COPY --from=build /app/dist /app/dist

# Expose the port that the server will run on
EXPOSE 5173

# Start serving the application
CMD ["serve", "-s", "dist", "-l", "5173"]