Test Cases for Recognition using Bayesian Network:

Test Case 0: (Equal probabilities)
	All conditions (G, A, H, T) are the same in the database. Face recognition confidences are 1/num_people for each person. Num_people is 5 (random number, can be increased to test how the system is for larger numbers).
	Result: unknown

Test Case 1: (Face recognition)
	All conditions (G, A, H, T) are the same in the database.
	
	a - Person A has 1.0 in face recognition confidences. The remaining confidences are 1/num_people for each person.
	Result: A

	b - Person A has 0.6 in face recognition confidences. The remaining confidences are 1/num_people for each person.
	Result: A

	c - Person A has 0.3 in face recognition confidences. The remaining confidences are 1/num_people for each person.
	Result: unknown

Test Case 2: (Gender recognition)
	A, H, T are the same in the database. Person B is female, all others are male.
	
	a - [1.0, "Female"] in gender recognition results.
	Result: B

	b - [0.6, "Female"] in gender recognition results.
	Result: B

	c - [0.8, "Male"] in gender recognition results.
	Result: unknown


Test Case 3: (Age recognition)
	G, H, T are the same in the database. Person C is 65 years old, all others are 33.
	
	a - [65, 1.0] in age recognition results.
	Result: C

	b - [65, 0.3] in age recognition results.
	Result: C

Test Case 4: (Height recognition)
	G, A, T are the same in the database. Person D is 205 cm tall, all others are 170.
	
	a - [205, 0.08] in height recognition results.
	Result: D

Test Case 5: (Time recognition)
	G, A, H are the same in the database. Person E was seen on Friday at the same time as others which were all seen on Monday.
	
	a - At the same time as E was seen on Friday.
	Result: E

	b - 15 min later than the time E was seen on Friday
	Result: E

	c- 8 hours before E was seen on Friday.
	Result: unknown

	d- Wednesday same time everyone was seen.
	Result: unknown

Test Case 6: (Real data)


	
