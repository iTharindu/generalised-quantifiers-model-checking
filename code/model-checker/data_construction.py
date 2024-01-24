import matplotlib.pyplot as plt

import seaborn as sns

import random
import re
import time

import numpy as np
import pandas as pd

from itertools import chain, combinations
from tqdm import tqdm
from fractions import Fraction
from sklearn.model_selection import train_test_split






male_names = ['Liam', 'Noah', 'Oliver', 'William', 'Elijah', 'James', 'Benjamin', 'Lucas', 'Mason', 'Ethan', 'Alexander', 'Henry', 'Jacob', 'Michael', 'Daniel', 'Logan', 'Jackson', 'Sebastian', 'Jack', 'Aiden', 'Owen', 'Samuel', 'Matthew', 'Joseph', 'Levi', 'Mateo', 'David', 'John', 'Wyatt', 'Carter', 'Julian', 'Luke', 'Grayson', 'Isaac', 'Jayden', 'Theodore', 'Gabriel', 'Anthony', 'Dylan', 'Leo', 'Lincoln', 'Jaxon', 'Asher', 'Christopher', 'Josiah', 'Andrew', 'Thomas', 'Joshua', 'Ezra', 'Hudson', 'Charles', 'Caleb', 'Isaiah', 'Ryan', 'Nathan', 'Adrian', 'Christian', 'Maverick', 'Colton', 'Elias', 'Aaron', 'Eli', 'Landon', 'Jonathan', 'Nolan', 'Hunter', 'Cameron', 'Connor', 'Santiago', 'Jeremiah', 'Ezekiel', 'Angel', 'Roman', 'Easton', 'Miles', 'Robert', 'Jameson', 'Nicholas', 'Greyson', 'Cooper', 'Ian', 'Carson', 'Axel', 'Jaxson', 'Dominic', 'Leonardo', 'Luca', 'Austin', 'Jordan', 'Adam', 'Xavier', 'Jose', 'Jace', 'Everett', 'Declan', 'Evan', 'Kayden', 'Parker', 'Wesley', 'Kai', 'Brayden', 'Bryson', 'Weston', 'Jason', 'Emmett', 'Sawyer', 'Silas', 'Bennett', 'Brooks', 'Micah', 'Damian', 'Harrison', 'Waylon', 'Ayden', 'Vincent', 'Ryder', 'Kingston', 'Rowan', 'George', 'Luis', 'Chase', 'Cole', 'Nathaniel', 'Zachary', 'Ashton', 'Braxton', 'Gavin', 'Tyler', 'Diego', 'Bentley', 'Amir', 'Beau', 'Gael', 'Carlos', 'Ryker', 'Jasper', 'Max', 'Juan', 'Ivan', 'Brandon', 'Jonah', 'Giovanni', 'Kaiden', 'Myles', 'Calvin', 'Lorenzo', 'Maxwell', 'Jayce', 'Kevin', 'Legend', 'Tristan', 'Jesus', 'Jude', 'Zion', 'Justin', 'Maddox', 'Abel', 'King', 'Camden', 'Elliott', 'Malachi', 'Milo', 'Emmanuel', 'Karter', 'Rhett', 'Alex', 'August', 'River', 'Xander', 'Antonio', 'Brody', 'Finn', 'Elliot', 'Dean', 'Emiliano', 'Eric', 'Miguel', 'Arthur', 'Matteo', 'Graham', 'Alan', 'Nicolas', 'Blake', 'Thiago', 'Adriel', 'Victor', 'Joel', 'Timothy', 'Hayden', 'Judah', 'Abraham', 'Edward', 'Messiah', 'Zayden', 'Theo', 'Tucker', 'Grant', 'Richard', 'Alejandro', 'Steven', 'Jesse', 'Dawson', 'Bryce', 'Avery', 'Oscar', 'Patrick', 'Archer', 'Barrett', 'Leon', 'Colt', 'Charlie', 'Peter', 'Kaleb', 'Lukas', 'Beckett', 'Jeremy', 'Preston', 'Enzo', 'Luka', 'Andres', 'Marcus', 'Felix', 'Mark', 'Ace', 'Brantley', 'Atlas', 'Remington', 'Maximus', 'Matias', 'Walker', 'Kyrie', 'Griffin', 'Kenneth', 'Israel', 'Javier', 'Kyler', 'Jax', 'Amari', 'Zane', 'Emilio', 'Knox', 'Adonis', 'Aidan', 'Kaden', 'Paul', 'Omar', 'BrianPeter', 'Louis', 'Caden', 'Maximiliano', 'Holden', 'Paxton', 'Nash', 'Bradley', 'Bryan', 'Simon', 'Phoenix', 'Lane', 'Josue', 'Colin', 'Rafael', 'Kyle', 'Riley', 'Jorge', 'Beckham', 'Cayden', 'Jaden', 'Emerson', 'Ronan', 'Karson', 'Arlo', 'Tobias', 'Brady', 'Clayton', 'Francisco', 'Zander', 'Erick', 'Walter', 'Daxton', 'Martin', 'Damien', 'Dallas', 'Cody', 'Chance', 'Jensen', 'Finley', 'Jett', 'Corbin', 'Kash', 'Reid', 'Kameron', 'Andre', 'Gunner', 'Jake', 'Hayes', 'Manuel', 'Prince', 'Bodhi', 'Cohen', 'Sean', 'Khalil', 'Hendrix', 'Derek', 'Cristian', 'Cruz', 'Kairo', 'Dante', 'Atticus', 'Killian', 'Stephen', 'Orion', 'Malakai', 'Ali', 'Eduardo', 'Fernando', 'Anderson', 'Angelo', 'Spencer', 'Gideon', 'Mario', 'Titus', 'Travis', 'Rylan', 'Kayson', 'Ricardo', 'Tanner', 'Malcolm', 'Raymond', 'Odin', 'Cesar', 'Lennox', 'Joaquin', 'Kane', 'Wade', 'Muhammad', 'Iker', 'Jaylen', 'Crew', 'Zayn', 'Hector', 'Ellis', 'Leonel', 'Cairo', 'Garrett', 'Romeo', 'Dakota', 'Edwin', 'Warren', 'Julius', 'Major', 'Donovan', 'Caiden', 'Tyson', 'Nico', 'Sergio', 'Nasir', 'Rory', 'Devin', 'Jaiden', 'Jared', 'Kason', 'Malik', 'Jeffrey', 'Ismael', 'Elian', 'Marshall', 'Lawson', 'Desmond', 'Winston', 'Nehemiah', 'Ari', 'Conner', 'Jay', 'Kade', 'Andy', 'Johnny', 'Jayceon', 'Marco', 'Seth', 'Ibrahim', 'Raiden', 'Collin', 'Edgar', 'Erik', 'Troy', 'Clark', 'Jaxton', 'Johnathan', 'Gregory', 'Russell', 'Royce', 'Fabian', 'Ezequiel', 'Noel', 'Pablo', 'Cade', 'Pedro', 'Sullivan', 'Trevor', 'Reed', 'Quinn', 'Frank', 'Harvey', 'Princeton', 'Zayne', 'Matthias', 'Conor', 'Sterling', 'Dax', 'Grady', 'Cyrus', 'Gage', 'Leland', 'Solomon', 'Emanuel', 'Niko', 'Ruben', 'Kasen', 'Mathias', 'Kashton', 'Franklin', 'Remy', 'Shane', 'Kendrick', 'Shawn', 'Otto', 'Armani', 'Keegan', 'Finnegan', 'Memphis', 'Bowen', 'Dominick', 'Kolton', 'Jamison', 'Allen', 'Philip', 'Tate', 'Peyton', 'Jase', 'Oakley', 'Rhys', 'Kyson', 'Adan', 'Esteban', 'Dalton', 'Gianni', 'Callum', 'Sage', 'Alexis', 'Milan', 'Moises', 'Jonas', 'Uriel', 'Colson', 'Marcos', 'Zaiden', 'Hank', 'Damon', 'Hugo', 'Ronin', 'Royal', 'Kamden', 'Dexter', 'Luciano', 'Alonzo', 'Augustus', 'Kamari', 'Eden', 'Roberto', 'Baker', 'Bruce', 'Kian', 'Albert', 'Frederick', 'Mohamed', 'Abram', 'Omari', 'Porter', 'Enrique', 'Alijah', 'Francis', 'Leonidas', 'Zachariah', 'Landen', 'Wilder', 'Apollo', 'Santino', 'Tatum', 'Pierce', 'Forrest', 'Corey', 'Derrick', 'Isaias', 'Kaison', 'Kieran', 'Arjun', 'Gunnar', 'Rocco', 'Emmitt', 'Abdiel', 'Braylen', 'Maximilian', 'Skyler', 'Phillip', 'Benson', 'Cannon', 'Deacon', 'Dorian', 'Asa', 'Moses', 'Ayaan', 'Jayson', 'Raul', 'Briggs', 'Armando', 'Nikolai', 'Cassius', 'Drew', 'Rodrigo', 'Raphael', 'Danny', 'Conrad', 'Moshe', 'Zyaire', 'Julio', 'Casey', 'Ronald', 'Scott', 'Callan', 'Roland', 'Saul', 'Jalen', 'Brycen', 'Ryland', 'Lawrence', 'Davis', 'Rowen', 'Zain', 'Ermias', 'Jaime', 'Duke', 'Stetson', 'Alec', 'Yusuf', 'Case', 'Trenton', 'Callen', 'Ariel', 'Jasiah', 'Soren', 'Dennis', 'Donald', 'Keith', 'Izaiah', 'Lewis', 'Kylan', 'Kobe', 'Makai', 'Rayan', 'Ford', 'Zaire', 'Landyn', 'Roy', 'Bo', 'Chris', 'Jamari', 'Ares', 'Mohammad', 'Darius', 'Drake', 'Tripp', 'Marcelo', 'Samson', 'Dustin', 'Layton', 'Gerardo', 'Johan', 'Kaysen', 'Keaton', 'Reece', 'Chandler', 'Lucca', 'Mack', 'Baylor', 'Kannon', 'Marvin', 'Huxley', 'Nixon', 'Tony', 'Cason', 'Mauricio', 'Quentin', 'Edison', 'Quincy', 'Ahmed', 'Finnley', 'Justice', 'Taylor', 'Gustavo', 'Brock', 'Ahmad', 'Kyree', 'Arturo', 'Nikolas', 'Boston', 'Sincere', 'Alessandro', 'Braylon', 'Colby', 'Leonard', 'Ridge', 'Trey', 'Aden', 'Leandro', 'Sam', 'Uriah', 'Ty', 'Sylas', 'Axton', 'Issac', 'Fletcher', 'Julien', 'Wells', 'Alden', 'Vihaan', 'Jamir', 'Valentino', 'Shepherd', 'Keanu', 'Hezekiah', 'Lionel', 'Kohen', 'Zaid', 'Alberto', 'Neil', 'Denver', 'Aarav', 'Brendan', 'Dillon', 'Koda', 'Sutton', 'Kingsley', 'Sonny', 'Alfredo', 'Wilson', 'Harry', 'Jaziel', 'Salvador', 'Cullen', 'Hamza', 'Dariel', 'Rex', 'Zeke', 'Mohammed', 'Nelson', 'Boone', 'Ricky', 'Santana', 'Cayson', 'Lance', 'Raylan', 'Lucian', 'Eliel', 'Alvin', 'Jagger', 'Braden', 'Curtis', 'Mathew', 'Jimmy', 'Kareem', 'Archie', 'Amos', 'Quinton', 'Yosef', 'Bodie', 'Jerry', 'Langston', 'Axl', 'Stanley', 'Clay', 'Douglas', 'Layne', 'Titan', 'Tomas', 'Houston', 'Darren', 'Lachlan', 'Kase', 'Korbin', 'Leighton', 'Joziah', 'Samir', 'Watson', 'Colten', 'Roger', 'Shiloh', 'Tommy', 'Mitchell', 'Azariah', 'Noe', 'Talon', 'Deandre', 'Lochlan', 'Joe', 'Carmelo', 'Otis', 'Randy', 'Byron', 'Chaim', 'Lennon', 'Devon', 'Nathanael', 'Bruno', 'Aryan', 'Flynn', 'Vicente', 'Brixton', 'Kyro', 'Brennan', 'Casen', 'Kenzo', 'Orlando', 'Castiel', 'Rayden', 'Ben', 'Grey', 'Jedidiah', 'Tadeo', 'Morgan', 'Augustine', 'Mekhi', 'Abdullah', 'Ramon', 'Saint', 'Emery', 'Maurice', 'Jefferson', 'Maximo', 'Koa', 'Ray', 'Jamie', 'Eddie', 'Guillermo', 'Onyx', 'Thaddeus', 'Wayne', 'Hassan', 'Alonso', 'Dash', 'Elisha', 'Jaxxon', 'Rohan', 'Carl', 'Kelvin', 'Jon', 'Larry', 'Reese', 'Aldo', 'Marcel', 'Melvin', 'Yousef', 'Aron', 'Kace', 'Vincenzo', 'Kellan', 'Miller', 'Jakob', 'Reign', 'Kellen', 'Kristopher', 'Ernesto', 'Briar', 'Gary', 'Trace', 'Joey', 'Clyde', 'Enoch', 'Jaxx', 'Crosby', 'Magnus', 'Fisher', 'Jadiel', 'Bronson', 'Eugene', 'Lee', 'Brecken', 'Atreus', 'Madden', 'Khari', 'Caspian', 'Ishaan', 'Kristian', 'Westley', 'Hugh', 'Kamryn', 'Musa', 'Rey', 'Thatcher', 'Alfred', 'Emory', 'Kye', 'Reyansh', 'Yahir', 'Cain', 'Mordechai', 'Zayd', 'Demetrius', 'Harley', 'Felipe', 'Louie', 'Branson', 'Graysen', 'Allan', 'Kole', 'Harold', 'Alvaro', 'Harlan', 'Amias', 'Brett', 'Khalid', 'Misael', 'Westin', 'Zechariah', 'Aydin', 'Kaiser', 'Lian', 'Bryant', 'Junior', 'Legacy', 'Ulises', 'Bellamy', 'Brayan', 'Kody', 'Ledger', 'Eliseo', 'Gordon', 'London', 'Rocky', 'Valentin', 'Terry', 'Damari', 'Trent', 'Bentlee', 'Canaan', 'Gatlin', 'Kiaan', 'Franco', 'Eithan', 'Idris', 'Krew', 'Yehuda', 'Marlon', 'Rodney', 'Creed', 'Salvatore', 'Stefan', 'Tristen', 'Adrien', 'Jamal', 'Judson', 'Camilo', 'Kenny', 'Nova', 'Robin', 'Rudy', 'Van', 'Bjorn', 'Brodie', 'Mac', 'Jacoby', 'Sekani', 'Vivaan', 'Blaine', 'Ira', 'Ameer', 'Dominik', 'Alaric', 'Dane', 'Jeremias', 'Kyng', 'Reginald', 'Bobby', 'Kabir', 'Jairo', 'Alexzander', 'Benicio', 'Vance', 'Wallace', 'Zavier', 'Billy', 'Callahan', 'Dakari', 'Gerald', 'Turner', 'Bear', 'Jabari', 'Cory', 'Fox', 'Harlem', 'Jakari', 'Jeffery', 'Maxton', 'Ronnie', 'Yisroel', 'Zakai', 'Bridger', 'Remi', 'Arian', 'Blaze', 'Forest', 'Genesis', 'Jerome', 'Reuben', 'Wesson', 'Anders', 'Banks', 'Calum', 'Dayton', 'Kylen', 'Dangelo', 'Emir', 'Malakhi', 'Salem', 'Blaise', 'Tru', 'Boden', 'Kolten', 'Kylo', 'Aries', 'Henrik', 'Kalel', 'Landry', 'Marcellus', 'Zahir', 'Lyle', 'Dario', 'Rene', 'Terrance', 'Xzavier', 'Alfonso', 'Darian', 'Kylian', 'Maison', 'Foster', 'Keenan', 'Yahya', 'Heath', 'Javion', 'Jericho', 'Aziel', 'Darwin', 'Marquis', 'Mylo', 'Ambrose', 'Anakin', 'Jordy', 'Juelz', 'Toby', 'Yael', 'Azrael', 'Brentley', 'Tristian', 'Bode', 'Jovanni', 'Santos', 'Alistair', 'Braydon', 'Kamdyn', 'Marc', 'Mayson', 'Niklaus', 'Simeon', 'Colter', 'Davion', 'Leroy', 'Ayan', 'Dilan', 'Ephraim', 'Anson', 'Merrick', 'Wes', 'Will', 'Jaxen', 'Maxim', 'Howard', 'Jad', 'Jesiah', 'Ignacio', 'Zyon', 'Ahmir', 'Jair', 'Mustafa', 'Jermaine', 'Yadiel', 'Aayan', 'Dhruv', 'Seven']


female_names = ['Olivia', 'Emma', 'Ava', 'Sophia', 'Isabella', 'Charlotte', 'Amelia', 'Mia', 'Harper', 'Evelyn', 'Abigail', 'Emily', 'Ella', 'Elizabeth', 'Camila', 'Luna', 'Sofia', 'Avery', 'Mila', 'Aria', 'Scarlett', 'Penelope', 'Layla', 'Chloe', 'Victoria', 'Madison', 'Eleanor', 'Grace', 'Nora', 'Riley', 'Zoey', 'Hannah', 'Hazel', 'Lily', 'Ellie', 'Violet', 'Lillian', 'Zoe', 'Stella', 'Aurora', 'Natalie', 'Emilia', 'Everly', 'Leah', 'Aubrey', 'Willow', 'Addison', 'Lucy', 'Audrey', 'Bella', 'Nova', 'Brooklyn', 'Paisley', 'Savannah', 'Claire', 'Skylar', 'Isla', 'Genesis', 'Naomi', 'Elena', 'Caroline', 'Eliana', 'Anna', 'Maya', 'Valentina', 'Ruby', 'Kennedy', 'Ivy', 'Ariana', 'Aaliyah', 'Cora', 'Madelyn', 'Alice', 'Kinsley', 'Hailey', 'Gabriella', 'Allison', 'Gianna', 'Serenity', 'Samantha', 'Sarah', 'Autumn', 'Quinn', 'Eva', 'Piper', 'Sophie', 'Sadie', 'Delilah', 'Josephine', 'Nevaeh', 'Adeline', 'Arya', 'Emery', 'Lydia', 'Clara', 'Vivian', 'Madeline', 'Peyton', 'Julia', 'Rylee', 'Brielle', 'Reagan', 'Natalia', 'Jade', 'Athena', 'Maria', 'Leilani', 'Everleigh', 'Liliana', 'Melanie', 'Mackenzie', 'Hadley', 'Raelynn', 'Kaylee', 'Rose', 'Arianna', 'Isabelle', 'Melody', 'Eliza', 'Lyla', 'Katherine', 'Aubree', 'Adalynn', 'Kylie', 'Faith', 'Mary', 'Margaret', 'Ximena', 'Iris', 'Alexandra', 'Jasmine', 'Charlie', 'Amaya', 'Taylor', 'Isabel', 'Ashley', 'Khloe', 'Ryleigh', 'Alexa', 'Amara', 'Valeria', 'Andrea', 'Parker', 'Norah', 'Eden', 'Elliana', 'Brianna', 'Emersyn', 'Valerie', 'Anastasia', 'Eloise', 'Emerson', 'Cecilia', 'Remi', 'Josie', 'Alina', 'Reese', 'Bailey', 'Lucia', 'Adalyn', 'Molly', 'Ayla', 'Sara', 'Daisy', 'London', 'Jordyn', 'Esther', 'Genevieve', 'Harmony', 'Annabelle', 'Alyssa', 'Ariel', 'Aliyah', 'Londyn', 'Juliana', 'Morgan', 'Summer', 'Juliette', 'Trinity', 'Callie', 'Sienna', 'Blakely', 'Alaia', 'Kayla', 'Teagan', 'Alaina', 'Brynlee', 'Finley', 'Catalina', 'Sloane', 'Rachel', 'Lilly', 'Ember', 'Kimberly', 'Juniper', 'Sydney', 'Arabella', 'Gemma', 'Jocelyn', 'Freya', 'June', 'Lauren', 'Amy', 'Presley', 'Georgia', 'Journee', 'Elise', 'Rosalie', 'Ada', 'Laila', 'Brooke', 'Diana', 'Olive', 'River', 'Payton', 'Ariella', 'Daniela', 'Raegan', 'Alayna', 'Gracie', 'Mya', 'Blake', 'Noelle', 'Ana', 'Leila', 'Paige', 'Lila', 'Nicole', 'Rowan', 'Hope', 'Ruth', 'Alana', 'Selena', 'Marley', 'Kamila', 'Alexis', 'Mckenzie', 'Zara', 'Millie', 'Magnolia', 'Kali', 'Kehlani', 'Catherine', 'Maeve', 'Adelyn', 'Sawyer', 'Elsie', 'Lola', 'Jayla', 'Adriana', 'Journey', 'Vera', 'Aspen', 'Joanna', 'Alivia', 'Angela', 'Dakota', 'Camille', 'Nyla', 'Tessa', 'Brooklynn', 'Malia', 'Makayla', 'Rebecca', 'Fiona', 'Mariana', 'Lena', 'Julianna', 'Vanessa', 'Juliet', 'Camilla', 'Kendall', 'Harley', 'Cali', 'Evangeline', 'Mariah', 'Jane', 'Zuri', 'Elaina', 'Sage', 'Amira', 'Adaline', 'Lia', 'Charlee', 'Delaney', 'Lilah', 'Miriam', 'Angelina', 'Mckenna', 'Aniyah', 'Phoebe', 'Michelle', 'Thea', 'Hayden', 'Maggie', 'Lucille', 'Amiyah', 'Annie', 'Alexandria', 'Myla', 'Vivienne', 'Kiara', 'Alani', 'Margot', 'Adelaide', 'Briella', 'Brynn', 'Saylor', 'Destiny', 'Amari', 'Evelynn', 'Haven', 'Phoenix', 'Izabella', 'Kaia', 'Lilliana', 'Harlow', 'Alessandra', 'Madilyn', 'Nina', 'Logan', 'Adelynn', 'Amina', 'Kate', 'Fatima', 'Samara', 'Winter', 'Giselle', 'Evie', 'Arielle', 'Jessica', 'Talia', 'Leia', 'Gabriela', 'Gracelyn', 'Lexi', 'Laura', 'Makenzie', 'Melissa', 'Royalty', 'Rylie', 'Raelyn', 'Gabrielle', 'Paris', 'Daleyza', 'Joy', 'Maisie', 'Oakley', 'Ariyah', 'Kailani', 'Alayah', 'Stephanie', 'Amora', 'Willa', 'Gracelynn', 'Elle', 'Keira', 'Tatum', 'Veronica', 'Milani', 'Felicity', 'Paislee', 'Allie', 'Nylah', 'Ariah', 'Cassidy', 'Lyric', 'Madeleine', 'Miracle', 'Gwendolyn', 'Octavia', 'Dahlia', 'Heidi', 'Celeste', 'Remington', 'Makenna', 'Everlee', 'Scarlet', 'Esmeralda', 'Maci', 'Lainey', 'Jacqueline', 'Kira', 'Lana', 'Brinley', 'Demi', 'Ophelia', 'Lennon', 'Reign', 'Bristol', 'Sabrina', 'Alaya', 'Jennifer', 'Kenzie', 'Angel', 'Luciana', 'Anaya', 'Hallie', 'Ryan', 'Camryn', 'Kinley', 'Daniella', 'Lilith', 'Blair', 'Amanda', 'Collins', 'Jordan', 'Maliyah', 'Rosemary', 'Cataleya', 'Kaylani', 'Gia', 'Alison', 'Leighton', 'Nadia', 'Sutton', 'Carolina', 'Skye', 'Alicia', 'Regina', 'Viviana', 'Yaretzi', 'Heaven', 'Serena', 'Raven', 'Emely', 'Carmen', 'Wren', 'Helen', 'Charleigh', 'Danielle', 'Daphne', 'Esme', 'Nayeli', 'Maddison', 'Sarai', 'Dylan', 'Frances', 'Elisa', 'Mabel', 'Skyler', 'Jenna', 'Emelia', 'Kaitlyn', 'Miranda', 'Marlee', 'Matilda', 'Selah', 'Jolene', 'Wynter', 'Hattie', 'Bianca', 'Haley', 'Lorelei', 'Mira', 'Braelynn', 'Annalise', 'Madelynn', 'Katie', 'Palmer', 'Aylin', 'Elliott', 'Kyla', 'Rory', 'Avianna', 'Liana', 'Shiloh', 'Kalani', 'Jada', 'Kelsey', 'Elianna', 'Jimena', 'Kora', 'Kamryn', 'Ainsley', 'Averie', 'Kensley', 'Helena', 'Holly', 'Emory', 'Macie', 'Amber', 'Zariah', 'Erin', 'Eve', 'Kathryn', 'Renata', 'Kayleigh', 'Emmy', 'Celine', 'Francesca', 'Fernanda', 'April', 'Shelby', 'Poppy', 'Colette', 'Meadow', 'Nia', 'Sierra', 'Cheyenne', 'Edith', 'Oaklynn', 'Kennedi', 'Abby', 'Danna', 'Jazlyn', 'Alessia', 'Mikayla', 'Alondra', 'Addilyn', 'Leona', 'Mckinley', 'Carter', 'Maren', 'Sylvia', 'Alejandra', 'Ariya', 'Astrid', 'Adrianna', 'Charli', 'Imani', 'Maryam', 'Christina', 'Stevie', 'Maia', 'Adelina', 'Dream', 'Aisha', 'Alanna', 'Itzel', 'Azalea', 'Katelyn', 'Kylee', 'Leslie', 'Madilynn', 'Myra', 'Virginia', 'Remy', 'Hanna', 'Aleah', 'Jaliyah', 'Antonella', 'Aviana', 'Cameron', 'Chelsea', 'Cecelia', 'Alia', 'Mae', 'Cadence', 'Emberly', 'Charley', 'Janelle', 'Mallory', 'Kaliyah', 'Elaine', 'Gloria', 'Jayleen', 'Lorelai', 'Malaysia', 'Bethany', 'Briana', 'Beatrice', 'Dorothy', 'Rosie', 'Jemma', 'Noa', 'Carly', 'Mariam', 'Anne', 'Karina', 'Emmalyn', 'Ivory', 'Ivanna', 'Jamie', 'Kara', 'Aitana', 'Jayda', 'Justice', 'Meredith', 'Briar', 'Skyla', 'Khaleesi', 'Dayana', 'Julieta', 'Katalina', 'Kendra', 'Oaklyn', 'Ashlyn', 'Armani', 'Jazmin', 'Kyra', 'Angelica', 'Zahra', 'Dallas', 'Johanna', 'Elliot', 'Macy', 'Monroe', 'Kimber', 'Henley', 'Ari', 'Karsyn', 'Lyanna', 'Lilian', 'Amalia', 'Nola', 'Dior', 'Aleena', 'Megan', 'Michaela', 'Amirah', 'Cassandra', 'Melany', 'Legacy', 'Reyna', 'Alma', 'Emmie', 'Melina', 'Siena', 'Priscilla', 'Ashlynn', 'Savanna', 'Sloan', 'Tiana', 'Aubrie', 'Coraline', 'Reina', 'Allyson', 'Kaydence', 'Sasha', 'Julie', 'Alexia', 'Irene', 'Marilyn', 'Greta', 'Braelyn', 'Emerie', 'Lylah', 'Nalani', 'Monica', 'Aileen', 'Lauryn', 'Anahi', 'Aurelia', 'Kassidy', 'Rayna', 'Romina', 'Lillie', 'Marie', 'Rosa', 'Saige', 'Bonnie', 'Kelly', 'Xiomara', 'Annabella', 'Avah', 'Lacey', 'Anya', 'Liberty', 'Karen', 'Mercy', 'Zelda', 'Baylee', 'Chaya', 'Kenna', 'Roselyn', 'Liv', 'Mara', 'Ensley', 'Malani', 'Malaya', 'Hadassah', 'Lyra', 'Adley', 'Galilea', 'Jaylah', 'Karla', 'Nala', 'Opal', 'Aliza', 'Milena', 'Ailani', 'Louisa', 'Mina', 'Kairi', 'Clementine', 'Louise', 'Maleah', 'Janiyah', 'Marina', 'Anika', 'Julissa', 'Bailee', 'Hayley', 'Jessie', 'Laney', 'Eileen', 'Faye', 'Kynlee', 'Tiffany', 'Lara', 'Angie', 'Joelle', 'Rhea', 'Calliope', 'Jazmine', 'Amani', 'Haylee', 'Aliana', 'Leyla', 'Jolie', 'Kinslee', 'Ryann', 'Simone', 'Milan', 'Lennox', 'Treasure', 'Alora', 'Ellis', 'Rebekah', 'Mikaela', 'Lina', 'Harmoni', 'Yareli', 'Giuliana', 'Lea', 'Harlee', 'Elyse', 'Frida', 'Blaire', 'Aya', 'Laurel', 'Meghan', 'Pearl', 'Zaylee', 'Alena', 'Holland', 'Bria', 'Rayne', 'Bridget', 'Zariyah', 'Kori', 'Frankie', 'Clarissa', 'Brylee', 'Davina', 'Rivka', 'Cynthia', 'Zaria', 'Madalyn', 'Paula', 'Salem', 'Amelie', 'Madisyn', 'Vienna', 'Haisley', 'Ainhoa', 'Journi', 'Karter', 'Oaklee', 'Livia', 'Miley', 'Adele', 'Amaia', 'Yara', 'Averi', 'Emmeline', 'Kyleigh', 'Princess', 'Penny', 'Sariyah', 'Amayah', 'Crystal', 'Keyla', 'Lilyana', 'Linda', 'Aniya', 'Marianna', 'Alaiya', 'Noemi', 'Chanel', 'Estella', 'Isabela', 'Jillian', 'Kallie', 'Ellianna', 'Elsa', 'Itzayana', 'Zora', 'Estelle', 'Chana', 'Raina', 'Royal', 'Sunny', 'Estrella', 'Martha', 'Ellen', 'Kailey', 'Maxine', 'Clare', 'Teresa', 'Annika', 'Kamilah', 'Azariah', 'Della', 'Addyson', 'Kai', 'Lilianna', 'Tinsley', 'Yaritza', 'Navy', 'Winnie', 'Andi', 'Kamiyah', 'Waverly', 'Sky', 'Amaris', 'Ramona', 'Saoirse', 'Hana', 'Judith', 'Halle', 'Laylah', 'Novalee', 'Jaycee', 'Zaniyah', 'Alianna', 'Paulina', 'Jayde', 'Thalia', 'Giovanna', 'Gwen', 'Iliana', 'Elora', 'Ezra', 'Kaylie', 'Braylee', 'Mavis', 'Ellison', 'Margo', 'Mylah', 'Paisleigh', 'Analia', 'August', 'Brittany', 'Kaisley', 'Belen', 'Promise', 'Amiya', 'Dalary', 'Veda', 'Alisson', 'Keilani', 'Oakleigh', 'Guadalupe', 'Leanna', 'Rosalyn', 'Selene', 'Theodora', 'Kamari', 'Anais', 'Elodie', 'Celia', 'Dani', 'Hunter', 'Indie', 'Kenia', 'Nellie', 'Belle', 'Kataleya', 'Lexie', 'Miah', 'Rylan', 'Sylvie', 'Valery', 'Addilynn', 'Dulce', 'Marissa', 'Meilani', 'Natasha', 'Jaylee', 'Kimora', 'Raquel', 'Scarlette', 'Aliya', 'Nataly', 'Whitney', 'Corinne', 'Denver', 'Nathalie', 'Kiera', 'Milana', 'Vada', 'Violeta', 'Luz', 'Addisyn', 'Casey', 'Deborah', 'Tori', 'Zainab', 'Erika', 'Jenesis', 'Avalynn', 'Nancy', 'Emmalynn', 'Hadlee', 'Heavenly', 'Aubrielle', 'Elisabeth', 'Salma', 'Adalee', 'Landry', 'Malayah', 'Novah', 'Egypt', 'Ayleen', 'Blessing', 'Elina', 'Joyce', 'Myah', 'Zoie', 'Christine', 'Jaelynn', 'Persephone', 'Chandler', 'Emmaline', 'Paloma', 'Harleigh', 'Noor', 'Paola', 'India', 'Madalynn', 'Rosalee', 'Florence', 'Maliah', 'Flora', 'Luella', 'Patricia', 'Whitley', 'Carolyn', 'Kathleen', 'Keily', 'Kiana', 'Tenley', 'Alyson', 'Barbara', 'Dana', 'Yasmin', 'Bexley', 'Micah', 'Tatiana', 'Arden', 'Aubriella', 'Lindsey', 'Emani', 'Hailee', 'Lisa', 'Sevyn', 'Fallon', 'Magdalena', 'Tinley', 'Halo', 'Lailah', 'Arlette', 'Ansley', 'Esperanza', 'Cleo', 'Aila', 'Emerald', 'Jaelyn', 'Karlee', 'Kaya', 'Ingrid', 'Jewel', 'Emilee', 'Giana', 'Paityn', 'Zola', 'Amoura', 'Renee', 'Ann', 'Berkley', 'Harriet', 'Queen', 'Sariah', 'Beatrix', 'Sandra', 'Alannah', 'Austyn', 'Freyja', 'Kaylin', 'Samira', 'Taliyah', 'Hadleigh', 'Kaiya', 'Robin', 'Luisa', 'Zendaya', 'Ariadne', 'Dixie']

names = male_names + female_names


nouns = [
    "actors", "artists", "butlers", "crooks", "directors",
    "experts", "fishermen", "judges", "jurors", "painters", "musicians",
    "policemen", "firemen", "professors", "sheriffs", "soldiers", "students",
    "philosophers", "teachers", "tourists", "lawyers", "physicians", "engineers",
    "veterinarians", "dentists", "accountants", "technicians", "electricians",
    "psychologists", "physicists", "plumbers", "waiters", "mechanics", "cooks",
    "librarians", "hairdressers", "economists", "bartenders", "cashiers", "surgeons",
    "pilots", "butchers", "opticians", "athletes", "cleaners",
    "actuaries", "sailors", "therapists", "secret_agents", "animal_breeders", "air_traffic_controllers",
    "anthropologists", "animal_trainers", "allergists", "real_estate_agents", "archeologists",
    "astronomers", "athletic_trainers", "audiologists", "auditors", "bailiffs",
    "bakers", "barbers", "clerks", "cartographers", "chiropractors", "dancers", "epidemiologists",
    "farmers", "floral_designers", "foresters", "truck_drivers", "jewelers", "interior_designers",
    "machinists", "mathematicians", "secretaries", "photographers", "radio_announcers", "roofers",
    "pavers", "taxi_drivers", "historians", "poets", "stunt_performers", "monologists", "publishers",
    "scribes", "bloggers", "copy_editors", "ceos", "ticket_controllers", "station_masters", "surveyors",
    "drillers", "scholars", "quants", "cfos", "ctos", "cios", "computer_scientists", "prisoners",
    "guests", "visitors", "helpers", "breadwinners", "hosts", "ghosts", "playmakers", "scorers",
    "settlers", "reachers", "cynics", "witches", "captains", "business_analysts", "data_scientists",
    "traders", "principals", "ballerinas", "footballers", "cricketers", "tennis_players", "lecturers",
    "patients", "ai_scientists", "philosophers", "cyclists", "chess_players", "strategists",
    "scientists", "parents", "fbi_agents", "defenders" , "attackers", "warlords", "nlp_engineers",
    "grandmasters", "masters", "kings", "queens", "knights", "princes", "princesses", "babies", "adults",
    "advisors", "wrestlers", "fighters", "boxers", "beekeepers", "musicians", "dj_artists", "violinists",
    "conductors", "gymnasts"
]

class ConstructModel:

    def create_model(self, predicates, domain, min_assignment = 0, max_assignment = -1):
        if max_assignment == -1:
            max_assignment = len(domain)
        else :
            max_assignment = max_assignment

        model = {
            'domain' : domain,
            'predicates' : {}
        }

        for i in predicates:
            k = random.randint(min_assignment, max_assignment)

            list_ds = random.sample(model['domain'], k)
            model['predicates'][i] = list_ds

        return model


    def convert_nl(self, model):

        domain = model['domain']

        nl_model = ', '.join(domain[:-1])

        # Add the final name with 'and'
        nl_model += f', and {domain[-1]} are people in a group. '

        for pred in model['predicates']:
            names = model['predicates'][pred]

            if len(names) == 0:
                nl_model += f'There are no {pred} in the group. '
            elif len(names) == 1:
                nl_model += f'Only {names[0]} is a {pred}. '
            else :
                sentence = ', '.join(names[:-1])
                sentence += f', and {names[-1]} are {pred}. '

                nl_model += sentence

        return nl_model

    def convert_nl_Ps(self, model):
        domain = model['domain']

        nl_model = f'There are {len(domain)} elements in a set, namely '
        nl_model += ', '.join(domain[:-1])

        # Add the final name with 'and'
        nl_model += f', and {domain[-1]}.'

        for pred in model['predicates']:
            names = model['predicates'][pred]

        if len(names) == 0:
            nl_model += f'There are no {pred} in the set. '
        elif len(names) == 1:
            nl_model += f'Only {names[0]} is a {pred}. '
        else :
            sentence = ', '.join(names[:-1])
            sentence += f', and {names[-1]} are {pred}. '

            nl_model += sentence

        return nl_model


class SentenceFormulae:

    def create_formulae(self, predicates, quantifier, min_num, max_num, num_form = 'Natural'):

        preds = random.sample(predicates, 2)

        negs = {
            preds[0] : random.choice([True, False]),
            preds[1] : random.choice([True, False])
        }
        if num_form == 'Natural':

            num = random.randint(min_num, max_num)

        elif num_form == 'Frac':
            q = random.randint(min_num, max_num)
            p = random.randint(min_num, q)

            num = Fraction(p,q)

        elif num_form == 'percentage':
            num = random.randint(min_num, 100)

        formula = (quantifier, num, preds[0], preds[1], negs)

        return formula

    def create_sentence(self, formula):
        #('at most', 1, 'fbi_agents', 'surgeons', {'fbi_agents': True, 'surgeons': True})
        (quantifier, num, preds1, preds2, negs) = formula

        if negs[preds1] == False:
            noun1 = preds1
        else :
            noun1 = f'non-{preds1}'

        if negs[preds2] == False:
            noun2 = preds2

        else:
            noun2 = f'non-{preds2}'

        if quantifier in ['At least', 'At most', 'More than', 'Less than']:
            return f'{quantifier} {num} {noun1} are {noun2}'

        elif quantifier == 'Equal':
            return f'{num} {noun1} are {noun2}'

        elif quantifier in ['Most', 'Few', 'All', 'Some']:
            return f'{quantifier} {noun1} are {noun2}'




class ModelChecker:

    def model_check(self, model, formula, num_form = 'Natural'):

        (quantifier, num, preds1, preds2, negs) = formula

        if negs[preds1] == False and negs[preds2] == False:
            union = set(model['predicates'][preds1]).intersection(model['predicates'][preds2])

        elif negs[preds1] == False and negs[preds2] == True:
            union = set(model['predicates'][preds1]).intersection(set(model['domain']) - set(model['predicates'][preds2]))

        elif negs[preds1] == True and negs[preds2] == False:
            union = set(set(model['domain']) - set(model['predicates'][preds1])).intersection(model['predicates'][preds2])

        else :
            union = set(set(model['domain']) - set(model['predicates'][preds1])).intersection(set(model['domain']) - set(model['predicates'][preds2]))


        if quantifier in ['Most', 'Few', 'All'] or num_form != 'Natural':
            if negs[preds1] == False :
                set_A = set(model['predicates'][preds1])
            else :
                set_A = set(set(model['domain']) - set(model['predicates'][preds1]))

            if quantifier == 'Most':
                return len(union) > 0.5 * len(set_A), union, set_A
            elif quantifier == 'Few':
                return len(union) < 0.5 * len(set_A), union, set_A
            elif quantifier == 'All':
                if negs[preds2] == False :
                    set_B = set(model['predicates'][preds2])
                else :
                    set_B = set(set(model['domain']) - set(model['predicates'][preds2]))
                return set_A.issubset(set_B), set_B, set_A

        else :
            if quantifier == 'At least':
                return len(union) >= num, union, None
            elif quantifier == 'At most':
                return len(union) <= num, union, None
            elif quantifier == 'More than':
                return len(union) > num, union, None
            elif quantifier == 'Less than':
                return len(union) < num, union, None
            elif quantifier == 'Equal':
                return len(union) == num, union, None
            elif quantifier == 'Some':
                return len(union) >= 1, union, None

class ConstructDataSet:

    def __init__(self, num_data_points, quantifier_list, min_dom, max_dom, min_pred, max_pred, boolean_coordinates = 0):
        self.num_data_points = num_data_points
        self.quantifier_list = quantifier_list
        self.min_dom = min_dom
        self.max_dom = max_dom
        self.min_pred = min_pred
        self.max_pred = max_pred
        self.boolean_coordinates = boolean_coordinates


    def generator(self):
        while True:
            yield

    def const_df(self, names, nouns, min_assignment, max_assignment, min_num, max_num):

        if self.boolean_coordinates == 0 :

            data = {
                'model': [],
                'model_nl': [],
                'formula': [],
                'sentence': [],
                'validity': [],
                'union': [],
                'set_subj':[],
                'quantifier': [],
                'num': [],
                'num_domain':[],
                'num_preds': [],
                'sub_neg': [],
                'nom_neg':[]
            }

        else :
            data = {
                'model': [],
                'model_nl': [],
                'formula': [],
                'sentence': [],
                'conjunction':[],
                'validity': [],
                'union': [],
                'set_subj':[],
                'quantifier': [],
                'num_domain':[],
                'num_preds': [],
                'valid_list': []
            }

        count = 0
        for quant in self.quantifier_list:
            time.sleep(0.1)

            for i in tqdm(self.generator()):

                Model = ConstructModel()

                num_domain = random.randint(self.min_dom, self.max_dom)
                num_preds = random.randint(self.min_pred, self.max_pred)


                max_assignment = num_domain
                max_num = num_domain

                domain = random.sample(names, num_domain)
                predicates = random.sample(nouns, num_preds)

                label = random.choice([True, False])

                Checker = ModelChecker()

                if self.boolean_coordinates == 0 :

                    sentenceFormula = SentenceFormulae()
                    formula = sentenceFormula.create_formulae(predicates, quantifier=quant, min_num=min_num, max_num=max_num)
                    sentence = sentenceFormula.create_sentence(formula)

                    interrupt = 0

                    while(True):

                        model = Model.create_model(predicates = predicates, domain=domain, min_assignment = min_assignment, max_assignment = max_assignment)
                        model_nl = Model.convert_nl_Ps(model)

                        validity, union, set_subj = Checker.model_check(model, formula)
                        interrupt += 1

                        if validity == label :

                            data['model'].append(model)
                            data['model_nl'].append(model_nl)
                            data['formula'].append(formula)
                            data['sentence'].append(sentence)
                            data['validity'].append(validity)
                            data['union'].append(union)
                            data['set_subj'].append(set_subj)
                            data['quantifier'].append(quant)
                            data['num'].append(formula[1])
                            data['num_domain'].append(len(domain))
                            data['num_preds'].append(len(predicates))
                            data['sub_neg'].append(int(formula[4][formula[2]]))
                            data['nom_neg'].append(int(formula[4][formula[3]]))

                            break

                        elif interrupt > 100 :
                            break

                else :

                    coordinate = random.choice(['and', 'or'])
                    sentenceFormula = SentenceFormulae()

                    sentence = []
                    formula = []


                    for i in range(self.boolean_coordinates+1):

                        formula_ = sentenceFormula.create_formulae(predicates, quantifier=quant, min_num=min_num, max_num=max_num)
                        sentence_  = sentenceFormula.create_sentence(formula_)

                        sentence.append(sentence_)
                        formula.append(formula_)

                    sentence = f' {coordinate} '.join(sentence)

                    interrupt = 0

                    while(True):

                        validity = []
                        set_subj = []
                        union = []

                        model = Model.create_model(predicates = predicates, domain=domain, min_assignment = min_assignment, max_assignment = max_assignment)
                        model_nl = Model.convert_nl(model)

                        for f in formula :
                            validity_, union_, set_subj_ = Checker.model_check(model, f)

                            union.append(union_)
                            set_subj.append(set_subj_)
                            validity.append(validity_)

                        if coordinate == 'and':
                            val_ = all(val == True for val in validity)
                        else :
                            val_ = any(val == True for val in validity)


                        interrupt += 1

                        if val_ == label :

                            data['model'].append(model)
                            data['model_nl'].append(model_nl)
                            data['formula'].append(formula)
                            data['sentence'].append(sentence)
                            data['validity'].append(val_)
                            data['union'].append(union)
                            data['valid_list'].append(validity)
                            data['set_subj'].append(set_subj)
                            data['quantifier'].append(quant)
                            data['num_domain'].append(len(domain))
                            data['num_preds'].append(len(predicates))
                            data['conjunction'].append(coordinate)

                            break

                        elif interrupt > 1000 :
                            break

                count += 1
                if count >= self.num_data_points/len(self.quantifier_list):
                    count = 0
                    break


        return pd.DataFrame(data)
  


if __name__ == "__main__":

    #args = parse_args()
    Data = ConstructDataSet(9000, ['At most', 'At least', 'Equal', 'More than', 'Less than', 'Most', 'Few', 'All', 'Some'], min_dom = 8, max_dom = 14, min_pred = 5, max_pred = 10, boolean_coordinates=0)
    df = Data.const_df(names, nouns, min_assignment=0, max_assignment=14, min_num = 1, max_num = 14)

    df.to_csv('output.csv', index = False)