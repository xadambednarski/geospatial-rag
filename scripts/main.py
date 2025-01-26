from agent import GeoSpatialAgent


def run(prompt: str):
    agent = GeoSpatialAgent(verbose=True).agent
    result = agent.invoke(prompt)
    return result


run("Potrzebuję dobrego masażu tajskiego w okolicy Placu Jana Pawła we Wrocławiu za mniej niż 300 złotych")
