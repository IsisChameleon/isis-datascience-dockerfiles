## How to run Neo4j as a container
https://neo4j.com/developer/docker-run-neo4j/


<!-- docker run --name isa-neo4j --network isa-network -p7474:7474 -p7687:7687 -d -v F:/Data/neo4j/data:/data -v F:/Data/neoj4/logs:/logs -v F:/Data/neoj4/import:/var/lib/neo4j/import -v F:/Data/neoj4/plugins:/plugins --env NEO4J_AUTH=neo4j/isabelle --env NEO4J_dbms_connector_https_advertised__address=localhost:7473 --env NEO4J_dbms_connector_http_advertised__address=localhost:7474 --env NEO4J_dbms_connector_bolt_advertised__address=localhost:7687 --env NEO4J_dbms_connector_bolt_address=0.0.0.0:7687  neo4j:latest  -->

docker run --name isa-neo4j --network isa-network  -p7474:7474 -p7687:7687 -d -v F:/Data/neo4j/data:/data -v F:/Data/neoj4/logs:/logs -v F:/Data/neoj4/import:/var/lib/neo4j/import -v F:/Data/neoj4/plugins:/plugins --env NEO4J_AUTH=neo4j/isabelle neo4j:latest 

https://neo4j.com/docs/operations-manual/current/docker/configuration/

Set environment variables for altering configurations
Defaults are set for many Neo4j configurations, such as pagecache and memory (512M each default). To change any configurations, we can use the --env parameter in our docker run command to set different values for the settings we want to change. Note: dot characters (.) become underscores (_) and underscores become double underscores (__).

we can set the password for the Docker container directly by specifying --env NEO4J_AUTH=neo4j/<password> in the run directive. We could also disable authentication entirely by specifying --env NEO4J_AUTH=none instead.

Another way is to run Neo4j as a non-root user by altering the docker run command with a different option. Instead of the --env, we can use the --user option and pass in the user’s id and group for access. 

docker run \
    ... \
    --user="$(id -u):$(id -g)" \
    neo4j:latest

## connect to your local instance using Neo4j browser
http://localhost:7474/browser/

## Running Cyphershell in container
Cypher Shell
If we want to run Cypher directly in our container, we need to first access our container.

docker exec -it isa-neo4j bash

cypher-shell -u neo4j -p isabelle (replace with correct user/password)

## Cypher doc
https://neo4j.com/docs/cypher-manual/current/
