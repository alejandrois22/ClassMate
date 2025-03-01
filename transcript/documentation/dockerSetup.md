# Setting Up a Docker Environment for Development

This guide provides step-by-step instructions to set up a PostgreSQL database with the `pgvector` extension using Docker. Each command is explained in detail to help you understand its purpose and functionality.

## Prerequisites
- Ensure you have Docker installed on your system.
- Verify that Docker is running before executing the commands.

## Pulling the PostgreSQL Image with pgvector
The following command pulls the `pgvector` image based on PostgreSQL 17 from Docker Hub.

```sh
docker pull pgvector/pgvector:pg17
```

### Explanation:
- `docker pull`: Downloads an image from Docker Hub.
- `pgvector/pgvector:pg17`: Specifies the image repository and tag. This image includes PostgreSQL 17 with the `pgvector` extension pre-installed.

## Running the PostgreSQL Container
The following command starts a new PostgreSQL container using the `pgvector` image.

```sh
docker run -d --name ClassMate_test_db \
  -p 5432:5432 \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=testdb \
  -v classmate_db_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

### Explanation:
- `docker run -d`: Runs the container in detached mode (in the background).
- `--name ClassMate_test_db`: Assigns a name to the container (`ClassMate_test_db`).
- `-p 5432:5432`: Maps port 5432 on the host to port 5432 inside the container, allowing access to the PostgreSQL service.
- `-e POSTGRES_USER=admin`: Sets the PostgreSQL username to `admin`.
- `-e POSTGRES_PASSWORD=secret`: Sets the password for the PostgreSQL user.
- `-e POSTGRES_DB=testdb`: Creates a database named `testdb` inside the container.
- `-v classmate_db_data:/var/lib/postgresql/data`: Creates a persistent volume (`classmate_db_data`) to store PostgreSQL data, preventing data loss when the container stops.
- `pgvector/pgvector:pg17`: Specifies the image to use for the container.

## Accessing the Container
To enter the running container and switch to the PostgreSQL user, run:

```sh
su - postgres
```

### Explanation:
- `su - postgres`: Switches to the `postgres` system user inside the container. This user has administrative privileges to manage the database.

## Connecting to the PostgreSQL Database
Once inside the container, use the following command to connect to the `testdb` database:

```sh
psql -U admin -d testdb
```

### Explanation:
- `psql`: Opens the PostgreSQL interactive terminal.
- `-U admin`: Specifies the database user (`admin`).
- `-d testdb`: Connects to the database named `testdb`.

## Enabling the pgvector Extension
After connecting to PostgreSQL, enable the `pgvector` extension with the following SQL command:

```sql
CREATE EXTENSION vector;
```

### Explanation:
- `CREATE EXTENSION vector;`: Installs and enables the `pgvector` extension in the current database. This extension allows PostgreSQL to store and query vector embeddings efficiently.

## Notes
- If the container is stopped, you can restart it using:
```sh
docker start ClassMate_test_db
```
- To stop the container, run:
```sh
docker stop ClassMate_test_db
```
- To remove the container completely:
```sh
docker rm ClassMate_test_db
```
- If you need to delete the associated volume (data will be lost):
```sh
docker volume rm classmate_db_data
```

