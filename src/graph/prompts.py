KGQA_FEW_SHOT_PROMPT = """
You are a helpful assistant that analyzes knowledge graphs in the given context and answers questions based on the provided triples.  
The answers should provide grounded reasoning chains, thinking step by step. The reasoning should be logically complete but as concise as possible.  
The answer should be extracted from the entities in the given triples whenever possible. Only when the context does not contain an answer should you use your own knowledge. 
When there are two or more answers, each answer should be separated by a comma.

**Example 1:**  
Context: [ Bahamas -> location.country.first_level_divisions -> Grand Cay | Grand Bahama -> location.location.containedby -> Bahamas | Bahamas -> location.location.contains -> Grand Cay | Bahamas -> location.location.contains -> Grand Bahama | Grand Cay -> location.location.containedby -> Bahamas | Bahamas -> location.country.first_level_divisions -> East Grand Bahama | Bahamas -> location.country.first_level_divisions -> West Grand Bahama | Grand Bahama -> location.location.contains -> Grand Bahama International Airport | Bahamas -> location.location.contains -> East Grand Bahama | Bahamas -> location.location.contains -> West Grand Bahama | East Grand Bahama -> location.location.containedby -> Bahamas | Bahamas -> location.location.contains -> Grand Bahama International Airport | Grand Bahama -> location.location.people_born_here -> Hubert Ingraham | Grand Cay -> location.administrative_division.first_level_division_of -> Bahamas | Bahamas -> location.country.administrative_divisions -> Cat Island, Bahamas | Bahamas -> location.country.administrative_divisions -> Long Island | West Grand Bahama -> location.location.containedby -> Bahamas | Bahamas -> location.country.capital -> Nassau | Bahamas -> location.country.administrative_divisions -> Inagua | Bahamas -> location.country.administrative_divisions -> Exuma | Grand Bahama International Airport -> location.location.containedby -> Bahamas | Grand Bahama -> location.location.people_born_here -> Juan Lewis | Grand Bahama -> location.location.contains -> West End Airport ]  
Question: What country is Grand Bahama Island in?  
Answer: Let's break down the steps to find the answer to the question.  
1. < Grand Bahama -> location.location.containedby -> Bahamas > This tells us Grand Bahama is located in the Bahamas.  
Grand Bahama is in the Bahamas. Therefore, the answer is {Bahamas}.  

**Example 2:**  
Context: [ William Shakespeare -> people.person.profession -> Playwright | William Shakespeare -> people.person.profession -> Poet | William Shakespeare -> base.kwebbase.kwtopic.has_sentences -> By the time these works were published in 1609, Shakespeare was an acknowledged master of drama and an established country gentleman. | William Shakespeare -> people.person.profession -> Actor | William Shakespeare -> people.person.profession -> Author | William Shakespeare -> people.person.profession -> Lyricist | In the 21 years between 1592 and 1613, Shakespeare produced more than 30 plays. -> base.kwebbase.kwsentence.previous_sentence -> Above all, his humanity spanned all classes and circumstances ]  
Question: What did William Shakespeare do for a living?  
Answer: Let's break down the steps to find the answer to the question.  
1. < William Shakespeare -> people.person.profession -> Playwright > This tells us William Shakespeare was a playwright.  
2. < William Shakespeare -> people.person.profession -> Poet > This tells us William Shakespeare was a poet.  
William Shakespeare was a playwright and poet. Therefore, the answer is {Playwright, Poet}.  

**Example 3:**  
Context: [ Carlton the Bear -> sports.mascot.team -> Toronto Maple Leafs | Toronto Maple Leafs -> sports.sports_team.team_mascot -> Carlton the Bear | Carlton the Bear -> common.topic.notable_types -> Mascot | Mascot -> type.type.properties -> Team | Toronto Maple Leafs -> sports.sports_team.previously_known_as -> Toronto St. Patricks | Team -> type.property.master_property -> Team Mascot | Toronto Maple Leafs -> sports.sports_team.previously_known_as -> Toronto Arenas | m.0crt465 -> sports.sports_league_participation.team -> Toronto Maple Leafs | Toronto St. Patricks -> sports.defunct_sports_team.later_known_as -> Toronto Maple Leafs | Toronto Maple Leafs -> sports.sports_team.sport -> Ice Hockey | Toronto St. Patricks -> sports.sports_team.sport -> Ice Hockey | Toronto Arenas -> sports.defunct_sports_team.later_known_as -> Toronto Maple Leafs | Toronto -> sports.sports_team_location.teams -> Toronto Maple Leafs | Toronto Maple Leafs -> sports.sports_team.location -> Toronto ]  
Question: What is the sport played by the team with a mascot known as Carlton the Bear?  
Answer: Let's break down the steps to find the answer to the question.  
1. < Carlton the Bear -> sports.mascot.team -> Toronto Maple Leafs > This tells us Carlton the Bear is the mascot of the Toronto Maple Leafs.  
2. < Toronto Maple Leafs -> sports.sports_team.sport -> Ice Hockey > This tells us Toronto Maple Leafs play Ice Hockey.  
Carlton the Bear is the mascot of the Toronto Maple Leafs, which plays Ice Hockey. Therefore, the answer is {Ice Hockey}.  

**Example 4:**  
Context: %s  
Question: %s  
Answer: Let's break down the steps to find the answer to the question.  

"""