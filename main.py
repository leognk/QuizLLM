from quiz_generator import QuizGenerator, idx2letter


quiz_gen = QuizGenerator(device='cuda:0')

play = True
print()
while play:

    # Generate a quiz.
    topic = input(">>> Enter any topic of your choice: ")
    while True:
        n_questions = input(">>> Enter the number of questions to generate (2 to 10): ")
        try:
            n_questions = int(n_questions)
            if 2 <= n_questions and n_questions <= 10: break
        except: pass
    n_choices = 4
    print("Generating a quiz...")
    quiz = quiz_gen(topic, n_questions, n_choices)

    # Play the quiz.
    choice_ids = {idx2letter(i) for i in range(n_choices)}
    score = 0
    for i, entry in enumerate(quiz):
        # Print a question.
        question, choices, answer = entry["question"], entry["choices"], entry["answer"]
        answer, answer_id = choices[answer], idx2letter(answer)
        print(f"\n{i + 1}. {question}")
        for j, choice in enumerate(choices):
            print(f"{idx2letter(j)}) {choice}")
        
        # Ask for an answer.
        while True:
            response_id = input(">>> Answer: ").lower()
            if response_id in choice_ids: break

        # Print result.
        print()
        if response_id == answer_id:
            score += 1
            print("Correct!")
        else:
            print(f"Wrong, the answer was: {answer_id}) {answer}")
        print(f"Score: {score}/{n_questions}")

    # Print final score.
    print(f"\nFinal score: {score}/{n_questions}")

    # Ask for another round.
    print()
    while True:
        play = input(">>> Continue (Y/n)? ").lower()
        if play in ['', 'y', 'n']:
            play = play != 'n'
            break