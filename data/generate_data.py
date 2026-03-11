"""
Generate synthetic MovieLens-style dataset for the recommendation system.
Falls back to this when OMDB API is not available or for ratings data.
"""

import pandas as pd
import numpy as np

# Curated movie list with rich metadata
MOVIES = [
    {"id": 1, "title": "The Shawshank Redemption", "year": 1994, "genres": "Drama", "director": "Frank Darabont", "actors": "Tim Robbins Morgan Freeman Bob Gunton William Sadler", "plot": "Two imprisoned men bond over a number of years finding solace and eventual redemption through acts of common decency"},
    {"id": 2, "title": "The Godfather", "year": 1972, "genres": "Crime Drama", "director": "Francis Ford Coppola", "actors": "Marlon Brando Al Pacino James Caan Robert Duvall", "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son"},
    {"id": 3, "title": "The Dark Knight", "year": 2008, "genres": "Action Crime Drama", "director": "Christopher Nolan", "actors": "Christian Bale Heath Ledger Aaron Eckhart Michael Caine", "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham Batman must accept one of the greatest psychological and physical tests"},
    {"id": 4, "title": "Pulp Fiction", "year": 1994, "genres": "Crime Drama Thriller", "director": "Quentin Tarantino", "actors": "John Travolta Uma Thurman Samuel L. Jackson Bruce Willis", "plot": "The lives of two mob hitmen a boxer a gangster and his wife and a pair of diner bandits intertwine in four tales of violence and redemption"},
    {"id": 5, "title": "Schindler's List", "year": 1993, "genres": "Biography Drama History", "director": "Steven Spielberg", "actors": "Liam Neeson Ralph Fiennes Ben Kingsley Caroline Goodall", "plot": "In German-occupied Poland during World War II industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution"},
    {"id": 6, "title": "Inception", "year": 2010, "genres": "Action Adventure Sci-Fi Thriller", "director": "Christopher Nolan", "actors": "Leonardo DiCaprio Joseph Gordon-Levitt Elliot Page Ken Watanabe", "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
    {"id": 7, "title": "The Matrix", "year": 1999, "genres": "Action Sci-Fi", "director": "Lana Wachowski", "actors": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss Hugo Weaving", "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers"},
    {"id": 8, "title": "Goodfellas", "year": 1990, "genres": "Biography Crime Drama", "director": "Martin Scorsese", "actors": "Ray Liotta Robert De Niro Joe Pesci Lorraine Bracco", "plot": "The story of Henry Hill and his life in the mob covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito"},
    {"id": 9, "title": "Fight Club", "year": 1999, "genres": "Drama Thriller", "director": "David Fincher", "actors": "Brad Pitt Edward Norton Helena Bonham Carter Meat Loaf", "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more"},
    {"id": 10, "title": "Interstellar", "year": 2014, "genres": "Adventure Drama Sci-Fi", "director": "Christopher Nolan", "actors": "Matthew McConaughey Anne Hathaway Jessica Chastain Michael Caine", "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival"},
    {"id": 11, "title": "The Silence of the Lambs", "year": 1991, "genres": "Crime Drama Thriller", "director": "Jonathan Demme", "actors": "Jodie Foster Anthony Hopkins Lawrence A. Bonney Kasi Lemmons", "plot": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer"},
    {"id": 12, "title": "The Lord of the Rings: The Return of the King", "year": 2003, "genres": "Adventure Drama Fantasy", "director": "Peter Jackson", "actors": "Elijah Wood Viggo Mortensen Ian McKellen Orlando Bloom", "plot": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring"},
    {"id": 13, "title": "Forrest Gump", "year": 1994, "genres": "Drama Romance", "director": "Robert Zemeckis", "actors": "Tom Hanks Robin Wright Gary Sinise Sally Field", "plot": "The presidencies of Kennedy and Johnson the Vietnam War the Watergate scandal and other historical events unfold from the perspective of an Alabama man"},
    {"id": 14, "title": "The Lion King", "year": 1994, "genres": "Animation Adventure Drama", "director": "Roger Allers Rob Minkoff", "actors": "Matthew Broderick Jeremy Irons James Earl Jones Moira Kelly", "plot": "Lion prince Simba and his father are targeted by his bitter uncle who wants to ascend the throne himself"},
    {"id": 15, "title": "Se7en", "year": 1995, "genres": "Crime Drama Mystery Thriller", "director": "David Fincher", "actors": "Morgan Freeman Brad Pitt Kevin Spacey Andrew Kevin Walker", "plot": "Two detectives a rookie and a veteran hunt a serial killer who uses the seven deadly sins as his motives"},
    {"id": 16, "title": "The Usual Suspects", "year": 1995, "genres": "Crime Drama Mystery Thriller", "director": "Bryan Singer", "actors": "Kevin Spacey Gabriel Byrne Chazz Palminteri Stephen Baldwin", "plot": "A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat which began when five criminals met at a seemingly random police lineup"},
    {"id": 17, "title": "Saving Private Ryan", "year": 1998, "genres": "Drama History War", "director": "Steven Spielberg", "actors": "Tom Hanks Tom Sizemore Edward Burns Barry Pepper", "plot": "Following the Normandy Landings a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action"},
    {"id": 18, "title": "The Green Mile", "year": 1999, "genres": "Crime Drama Fantasy Mystery", "director": "Frank Darabont", "actors": "Tom Hanks David Morse Michael Clarke Duncan Bonnie Hunt", "plot": "The lives of guards on Death Row are affected by one of their charges a black man accused of child murder and rape yet who has a mysterious gift"},
    {"id": 19, "title": "Gladiator", "year": 2000, "genres": "Action Adventure Drama", "director": "Ridley Scott", "actors": "Russell Crowe Joaquin Phoenix Connie Nielsen Oliver Reed", "plot": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery"},
    {"id": 20, "title": "The Departed", "year": 2006, "genres": "Crime Drama Thriller", "director": "Martin Scorsese", "actors": "Leonardo DiCaprio Matt Damon Jack Nicholson Mark Wahlberg", "plot": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston"},
    {"id": 21, "title": "Whiplash", "year": 2014, "genres": "Drama Music", "director": "Damien Chazelle", "actors": "Miles Teller J.K. Simmons Melissa Benoist Paul Reiser", "plot": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing"},
    {"id": 22, "title": "The Prestige", "year": 2006, "genres": "Drama Mystery Sci-Fi Thriller", "director": "Christopher Nolan", "actors": "Christian Bale Hugh Jackman Scarlett Johansson Michael Caine", "plot": "After a tragic accident two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other"},
    {"id": 23, "title": "Parasite", "year": 2019, "genres": "Comedy Drama Thriller", "director": "Bong Joon Ho", "actors": "Song Kang-ho Lee Sun-kyun Cho Yeo-jeong Choi Woo-shik", "plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan"},
    {"id": 24, "title": "Avengers: Endgame", "year": 2019, "genres": "Action Adventure Drama Sci-Fi", "director": "Anthony Russo Joe Russo", "actors": "Robert Downey Jr. Chris Evans Mark Ruffalo Chris Hemsworth", "plot": "After the devastating events of Infinity War the universe is in ruins with the help of remaining allies the Avengers assemble once more"},
    {"id": 25, "title": "Joker", "year": 2019, "genres": "Crime Drama Thriller", "director": "Todd Phillips", "actors": "Joaquin Phoenix Robert De Niro Zazie Beetz Frances Conroy", "plot": "In Gotham City mentally troubled comedian Arthur Fleck is disregarded and mistreated by society He then embarks on a downward spiral of revolution and bloody crime"},
    {"id": 26, "title": "The Wolf of Wall Street", "year": 2013, "genres": "Biography Comedy Crime Drama", "director": "Martin Scorsese", "actors": "Leonardo DiCaprio Jonah Hill Margot Robbie Matthew McConaughey", "plot": "Based on the true story of Jordan Belfort from his rise to a wealthy stock-broker living the high life to his fall involving crime corruption and the federal government"},
    {"id": 27, "title": "Django Unchained", "year": 2012, "genres": "Drama Western", "director": "Quentin Tarantino", "actors": "Jamie Foxx Christoph Waltz Leonardo DiCaprio Kerry Washington", "plot": "With the help of a German bounty-hunter a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner"},
    {"id": 28, "title": "Blade Runner 2049", "year": 2017, "genres": "Action Drama Mystery Sci-Fi Thriller", "director": "Denis Villeneuve", "actors": "Harrison Ford Ryan Gosling Ana de Armas Dave Bautista", "plot": "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard who's been missing for thirty years"},
    {"id": 29, "title": "Arrival", "year": 2016, "genres": "Drama Mystery Sci-Fi Thriller", "director": "Denis Villeneuve", "actors": "Amy Adams Jeremy Renner Forest Whitaker Michael Stuhlbarg", "plot": "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world"},
    {"id": 30, "title": "Dune", "year": 2021, "genres": "Action Adventure Drama Sci-Fi", "director": "Denis Villeneuve", "actors": "Timothée Chalamet Rebecca Ferguson Oscar Isaac Josh Brolin", "plot": "Feature adaptation of Frank Herbert's science fiction novel about the son of a noble family entrusted with the protection of the most valuable asset in the galaxy"},
    {"id": 31, "title": "Get Out", "year": 2017, "genres": "Horror Mystery Thriller", "director": "Jordan Peele", "actors": "Daniel Kaluuya Allison Williams Bradley Whitford Catherine Keener", "plot": "A young African-American visits his white girlfriend's parents for the weekend where his simmering uneasiness about their reception of him eventually reaches a boiling point"},
    {"id": 32, "title": "Hereditary", "year": 2018, "genres": "Drama Horror Mystery Thriller", "director": "Ari Aster", "actors": "Toni Collette Milly Shapiro Gabriel Byrne Alex Wolff", "plot": "A grieving family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother"},
    {"id": 33, "title": "La La Land", "year": 2016, "genres": "Comedy Drama Music Romance", "director": "Damien Chazelle", "actors": "Ryan Gosling Emma Stone John Legend Rosemarie DeWitt", "plot": "While navigating their careers in Los Angeles a pianist and an actress fall in love while attempting to reconcile their aspirations for the future"},
    {"id": 34, "title": "Mad Max: Fury Road", "year": 2015, "genres": "Action Adventure Sci-Fi Thriller", "director": "George Miller", "actors": "Tom Hardy Charlize Theron Nicholas Hoult Hugh Keays-Byrne", "plot": "In a post-apocalyptic wasteland a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners a psychotic worshiper and a drifter named Max"},
    {"id": 35, "title": "The Grand Budapest Hotel", "year": 2014, "genres": "Adventure Comedy Crime Drama Mystery Romance", "director": "Wes Anderson", "actors": "Ralph Fiennes F. Murray Abraham Mathieu Amalric Adrien Brody", "plot": "A writer encounters the owner of an aging high-class hotel who tells him of his early years serving as a lobby boy in the hotel's glorious years"},
    {"id": 36, "title": "Spirited Away", "year": 2001, "genres": "Animation Adventure Family Fantasy", "director": "Hayao Miyazaki", "actors": "Daveigh Chase Suzanne Pleshette Miyu Irino Rumi Hiiragi", "plot": "During her family's move to the suburbs a sullen 10-year-old girl wanders into a world ruled by gods witches and spirits and where humans are changed into beasts"},
    {"id": 37, "title": "Oldboy", "year": 2003, "genres": "Action Drama Mystery Thriller", "director": "Park Chan-wook", "actors": "Choi Min-sik Yoo Ji-tae Kang Hye-jung Chi Dae-han", "plot": "After being kidnapped and imprisoned for fifteen years Oh Dae-Su is released only to find that he must find his captor in five days"},
    {"id": 38, "title": "The Truman Show", "year": 1998, "genres": "Comedy Drama Sci-Fi", "director": "Peter Weir", "actors": "Jim Carrey Laura Linney Noah Emmerich Ed Harris", "plot": "An insurance salesman discovers his whole life is actually a reality TV show"},
    {"id": 39, "title": "A Beautiful Mind", "year": 2001, "genres": "Biography Drama", "director": "Ron Howard", "actors": "Russell Crowe Ed Harris Jennifer Connelly Paul Bettany", "plot": "After John Nash a brilliant but asocial mathematician accepts secret work in cryptography his life takes a turn for the nightmarish"},
    {"id": 40, "title": "No Country for Old Men", "year": 2007, "genres": "Crime Drama Thriller Western", "director": "Joel Coen Ethan Coen", "actors": "Tommy Lee Jones Javier Bardem Josh Brolin Woody Harrelson", "plot": "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash near the Rio Grande"},
    {"id": 41, "title": "There Will Be Blood", "year": 2007, "genres": "Drama Western", "director": "Paul Thomas Anderson", "actors": "Daniel Day-Lewis Paul Dano Ciarán Hinds Martin Stringer", "plot": "A story of family religion hatred oil and madness focusing on a turn-of-the-century prospector in the early days of the Texas oil business"},
    {"id": 42, "title": "2001: A Space Odyssey", "year": 1968, "genres": "Mystery Sci-Fi", "director": "Stanley Kubrick", "actors": "Keir Dullea Gary Lockwood William Sylvester Daniel Richter", "plot": "After discovering a mysterious artifact buried beneath the Lunar surface mankind sets off on a quest to find its origins with help from intelligent supercomputer H.A.L. 9000"},
    {"id": 43, "title": "Apocalypse Now", "year": 1979, "genres": "Drama War", "director": "Francis Ford Coppola", "actors": "Martin Sheen Marlon Brando Robert Duvall Frederic Forrest", "plot": "A U.S. Army officer serving in Vietnam is tasked with assassinating a renegade Special Forces Colonel who sees himself as a god"},
    {"id": 44, "title": "Eternal Sunshine of the Spotless Mind", "year": 2004, "genres": "Drama Romance Sci-Fi", "director": "Michel Gondry", "actors": "Jim Carrey Kate Winslet Tom Wilkinson Gerry Robert Byrne", "plot": "When their relationship turns sour a couple undergoes a medical procedure to have each other erased from their memories"},
    {"id": 45, "title": "Her", "year": 2013, "genres": "Drama Romance Sci-Fi", "director": "Spike Jonze", "actors": "Joaquin Phoenix Amy Adams Scarlett Johansson Rooney Mara", "plot": "In a near future a lonely writer develops an unlikely relationship with an operating system designed to meet his every need"},
    {"id": 46, "title": "Memento", "year": 2000, "genres": "Mystery Thriller", "director": "Christopher Nolan", "actors": "Guy Pearce Carrie-Anne Moss Joe Pantoliano Mark Boone Junior", "plot": "A man with short-term memory loss attempts to track down his wife's murderer"},
    {"id": 47, "title": "Pan's Labyrinth", "year": 2006, "genres": "Drama Fantasy Thriller War", "director": "Guillermo del Toro", "actors": "Ivana Baquero Ariadna Gil Doug Jones Sergi López", "plot": "In the Falangist Spain of 1944 the bookish young stepdaughter of a sadistic army officer escapes into an eerie but captivating fantasy world"},
    {"id": 48, "title": "Mulholland Drive", "year": 2001, "genres": "Drama Mystery Thriller", "director": "David Lynch", "actors": "Naomi Watts Laura Harring Justin Theroux Ann Miller", "plot": "After a car wreck on the winding Mulholland Drive renders a woman amnesiac she and a perky Hollywood-hopeful search for clues and answers across Los Angeles"},
    {"id": 49, "title": "Black Swan", "year": 2010, "genres": "Drama Horror Mystery Thriller", "director": "Darren Aronofsky", "actors": "Natalie Portman Mila Kunis Vincent Cassel Barbara Hershey", "plot": "A committed dancer wins the lead role in a production of Tchaikovsky's Swan Lake only to find herself struggling to maintain her sanity"},
    {"id": 50, "title": "The Social Network", "year": 2010, "genres": "Biography Drama", "director": "David Fincher", "actors": "Jesse Eisenberg Andrew Garfield Justin Timberlake Rooney Mara", "plot": "As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook he is sued by the twins who claimed he stole their idea"},
]

def generate_ratings(n_users=200, n_movies=50, seed=42):
    """Generate synthetic user-movie ratings mimicking MovieLens format."""
    np.random.seed(seed)
    
    # Genre preference profiles for users
    genre_profiles = {
        "action_fan": ["Action", "Adventure", "Sci-Fi", "Thriller"],
        "drama_lover": ["Drama", "Biography", "History", "War"],
        "thriller_seeker": ["Crime", "Mystery", "Thriller", "Horror"],
        "sci_fi_geek": ["Sci-Fi", "Adventure", "Mystery", "Fantasy"],
        "classic_cinephile": ["Drama", "Crime", "Western", "War"],
        "romance_fan": ["Romance", "Drama", "Comedy", "Music"],
    }
    
    profile_names = list(genre_profiles.keys())
    user_profiles = np.random.choice(profile_names, n_users)
    
    ratings = []
    for user_id in range(1, n_users + 1):
        profile = user_profiles[user_id - 1]
        preferred_genres = genre_profiles[profile]
        
        # Each user rates between 10 and 30 movies
        n_ratings = np.random.randint(10, 30)
        movie_sample = np.random.choice(range(len(MOVIES)), n_ratings, replace=False)
        
        for movie_idx in movie_sample:
            movie = MOVIES[movie_idx]
            movie_genres = movie["genres"]
            
            # Base rating
            base_rating = 3.0
            
            # Boost if genre matches profile
            genre_match = sum(1 for g in preferred_genres if g in movie_genres)
            base_rating += genre_match * 0.4
            
            # Add noise
            noise = np.random.normal(0, 0.5)
            rating = np.clip(base_rating + noise, 1.0, 5.0)
            rating = round(rating * 2) / 2  # Round to nearest 0.5
            
            ratings.append({
                "userId": user_id,
                "movieId": movie["id"],
                "rating": rating
            })
    
    return pd.DataFrame(ratings)


def get_movies_df():
    """Return movies as a DataFrame."""
    return pd.DataFrame(MOVIES)


def get_ratings_df():
    """Return ratings as a DataFrame."""
    return generate_ratings()
