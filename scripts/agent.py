from langchain_openai import OpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
from geodata_api import reverse_geocode, calculate_distance
from utils import load_data
from langchain.agents import AgentType, AgentExecutor
import json


load_dotenv()


class GeoSpatialAgent:
    def __init__(self, temperature: float = 0, verbose: bool = False):
        global llm, data
        data = load_data()
        llm = OpenAI(temperature=temperature)
        self.tools = [
            Tool(
                "extract_intent",
                self.extract_intent,
                description="Extracts user intent and location from a query and returns a dictionary with service, max price, and location.",
            ),
            Tool(
                "find_businesses",
                self.find_businesses,
                description="Finds businesses based on services, prices, and ratings.",
            ),
        ]
        self.agent = self._initialize_agent(verbose=verbose)

    @staticmethod
    def generate_similar_services(user_service):
        """
        Generate a list of similar services based on the input service.
        """
        support_prompt = f"""
            Generate a list of 10 names of services that are similar to '{user_service}'.
            - Use lowercase only.
            - Return the result as a simple comma-separated list without any numbering or additional text.
            - Do not include explanations, code, or unrelated content.
            - Examples of expected output for similar requests:
                - Input: 'masaż tajski', Output: 'masaż tajski, masaż relaksacyjny, masaż leczniczy, masaż sportowy, masaż aromaterapeutyczny, masaż shiatsu, masaż lomi lomi, masaż głęboki, masaż limfatyczny, masaż szwedzki, masaż gorącymi kamieniami'
                - Input: 'fizjoterapia', Output: 'fizjoterapia, rehabilitacja, terapia manualna, kinezyterapia, osteopatia, masaż leczniczy, terapia bólu pleców, terapia ruchowa, elektroterapia, krioterapia, hydroterapia'

            Now, generate a similar list for: '{user_service}'.
        """

        response = llm.invoke(support_prompt)

        try:
            similar_services = [
                x.strip().replace("\n", "") for x in response.split("<|endoftext|>")[0].split(", ")
            ]
            similar_services = [user_service] + similar_services
        except Exception:
            similar_services = [user_service]

        return similar_services

    @staticmethod
    def filter_businesses(data, similar_services, max_price):
        """
        Filter the businesses based on the services, prices, and ratings.
        """
        relevant_businesses = []
        for business in data:
            for service in business["services"]:
                if service["price"] > max_price:
                    continue
                if service["name"].lower() in similar_services:
                    business["mss"] = service["name"]
                    business["mss_price"] = service["price"]
                    relevant_businesses.append(business)
                    break
        return relevant_businesses

    @staticmethod
    def geolocate_user(location):
        """
        Reverse geocode the user's location and return latitude and longitude.
        """
        try:
            user_geolocation = reverse_geocode(location.replace(" ", "+"))[0]
            user_lat, user_lon = user_geolocation["lat"], user_geolocation["lon"]
            return user_lat, user_lon
        except Exception:
            return None, None

    @staticmethod
    def calculate_distances(relevant_businesses, user_lat, user_lon):
        """
        Calculate the distances between the user and the businesses.
        """
        if user_lat and user_lon:
            for business in relevant_businesses:
                business["distance_km"] = calculate_distance(
                    user_lat,
                    user_lon,
                    business["geolocation"]["latitude"],
                    business["geolocation"]["longitude"],
                )
        else:
            for business in relevant_businesses:
                business["distance_km"] = float("inf")

        return relevant_businesses

    @staticmethod
    def format_businesses(relevant_businesses):
        """
        Format the top 5 businesses and return the result message.
        """
        top_relevant_businesses = sorted(
            relevant_businesses, key=lambda x: (x["distance_km"], -x["rating"])
        )[:5]

        if not top_relevant_businesses:
            return {"message": "No businesses found matching your criteria."}

        msg = "Here are the top 5 businesses that match your criteria:"
        for business in top_relevant_businesses:
            msg += f"- **{business['mss']}** at **{business['name']}** ({business['distance_km']:.2f} km away) for {business['mss_price']} zł. Rating: {business['rating']}⭐. Address: {business['address']}"
        return {"message": msg}

    @tool("extract_intent")
    def extract_intent(input_text) -> dict:
        """
        Extracts user intent and location from a query. Returns a dictionary with:
        - service: The service user is looking for (not a business name, but a service).
        - max_price: The maximum price (float, if mentioned, otherwise null).
        - location: User's location (simplified to street name, number and city name).
        Pass the input_text as a second parameter to the function, because the function is a method of the class and the first parameter is always self.
        """
        prompt = f"""
        You are an assistant designed to extract key information from user queries.
        Analyze the following query and return a JSON object with these exact fields:
        - "service" (string): The service the user is looking for. If user didn't specify a service but mentioned a business, return the service offered by the business.
        - "max_price" (float or null): The maximum price the user is willing to pay. If not mentioned, return null.
        - "location" (string): The user's location as a clear address or place. If unclear, return "unknown".

        Query: "{input_text}"

        Examples:
        1. Query: "Szukam masażu tajskiego za mniej niż 300zł, moja lokalizacja to Plac Grunwaldzki 21 Wrocław."
        Output: {{"service": "masaż tajski", "max_price": 300, "location": "Plac Grunwaldzki 21 Wrocław"}}
        2. Query: "Potrzebuję kogoś do naprawy mojego roweru, jestem w Krakowie."
        Output: {{"service": "naprawa roweru", "max_price": null, "location": "Kraków"}}
        3. Query: "Czy znajdzie się ktoś do sprzątania biura? Maksymalnie mogę zapłacić 150zł."
        Output: {{"service": "sprzątanie biura", "max_price": 150, "location": "unknown"}}
        4. Query: "Szukam fryzjera w okolicy ulicy Fabrycznej we Wrocławiu, który oferuje strzyżenie męskie za mniej niż 70 złotych."
        Output: {{"service": "strzyżenie męskie", "max_price": 70, "location": "Fabryczna Wrocław"}}

        Now extract the information:
        """
        response = llm.invoke(prompt)
        response = response.split("}")[0] + "}"

        try:
            extracted_data = json.loads(response)
            if all(key in extracted_data for key in ["service", "max_price", "location"]):
                return response
        except Exception:
            pass

        return {"service": "unknown", "max_price": None, "location": "unknown"}

    @tool("find_businesses")
    def find_businesses(input_data):
        """
        Finds businesses based on services, prices, and ratings. Returns a list of dictionaries with:
        - name: The name of the business.
        - address: The address of the business.
        - rating: The rating of the business (float).
        - service: The service offered by the business matching the user's query.
        - price: The price of the service offered by the business.

        The output should be displayed nicely, with each business on a new line, including all the information.
        """

        input_data = json.loads(input_data)
        user_service = input_data["service"].lower()
        location = input_data["location"]
        max_price = input_data["max_price"]

        similar_services = GeoSpatialAgent.generate_similar_services(user_service)

        relevant_businesses = GeoSpatialAgent.filter_businesses(data, similar_services, max_price)

        user_lat, user_lon = GeoSpatialAgent.geolocate_user(location)

        relevant_businesses = GeoSpatialAgent.calculate_distances(
            relevant_businesses, user_lat, user_lon
        )

        return GeoSpatialAgent.format_businesses(relevant_businesses)

    def _initialize_agent(
        self, agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    ) -> AgentExecutor:
        agent = initialize_agent(self.tools, llm, agent=agent_type, verbose=verbose)
        return agent
