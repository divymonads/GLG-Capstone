class Mapper(object):
    def __init__(self):
        self.list = [
        "Police, violence, and crime",
        "Social Media",
        "Law, Justice, and Immigration",
        "Technology Companies",
        "Times, Event, and Travel",
        "Military Operations and Disasters",
        "Entertainment",
        "Local, Housing, and City/Town Issues",
        "Fashion",
        "History, Art, and Education",
        "Sports",
        "Consumer and Retail Investing",
        "Political and Russian Investigations",
        "Health, Disease, and Pandemics",
        "Daily Living",
        "Gender and Racial Issues",
        "Energy and Climate Issues",
        "Domestic and Family",
        "US Electoral Politics",
        "Foreign Affairs",
        "Company Performance Expectations",
        "Bars, Restaurants, and Pets",
        "Stock Market",
        "Jobs and Careers",
        "Government Spending"
        ]

        self.SMEs = [
        "Eric Holder",
        "Cecilia Kang",
        "Jacqueline McKenzie",
        "Paul Graham",
        "Priya Parker",
        "Lucy Jones",
        "Don Franks",
        "Shaun Donovan",
        "Kim Kardashian",
        "Miguel Cardona",
        "Critiano Ronaldo",
        "Jim Cramer",
        "Robert Mueller",
        "Anthony Fauci",
        "Toni Morrison",
        "Elana Ruiz",
        "Jennifer Granholm",
        "Tom Daschle",
        "Stephen Ansolabehere",
        "William Burns",
        "Congsheng Wu",
        "Jonathan Gold",
        "George Bragues",
        "James Van",
        "Neera Tanden"
        ]

    def get(self, i):
        if 0 <= i < 25:
            return self.list[i]

        return "N/A"

    def getExpert(self, i):
        if 0 <= i < 25:
            return self.SMEs[i]

        return "N/A"
