from agent import GeoSpatialAgent


def run(prompt: str):
    agent = GeoSpatialAgent(verbose=True).agent
    result = agent.invoke(prompt)
    return result


run("Chciałbym ostrzyc brodę za mniej niż 60 złotych w okolicach Popowickiej we Wrocławiu")
