from app.services.index_service import load_query_engine


engine = load_query_engine()
response = engine.query("What controls should govern consequential agent actions?")
print(response)
print(response.source_nodes)