# ============================================================
# Examination Timetabling Problem (ETP)
# Metaheuristic Optimisation Methods
# Algorithms:
# 1. Genetic Algorithm (GA)
# 2. Simulated Annealing (SA)
# 3. Particle Swarm Optimisation (PSO)
# ============================================================

import random
import math
import numpy as np
import time
from copy import deepcopy

# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "HEC-S-92": {
        "exams": 81,
        "students": 2800,
        "slots": 18
    },
    "EAR-F-83": {
        "exams": 190,
        "students": 11000,
        "slots": 24
    },
    "CAR-S-91": {
        "exams": 682,
        "students": 18000,
        "slots": 35
    }
}

# ============================================================
# GENERATE CONFLICT MATRIX
# ============================================================

def generate_conflict_matrix(num_exams, density=0.05):

    matrix = np.zeros((num_exams, num_exams), dtype=int)

    for i in range(num_exams):
        for j in range(i + 1, num_exams):

            if random.random() < density:
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def calculate_penalty(timetable, conflicts, students):

    penalty = 0
    exams = len(timetable)

    for i in range(exams):
        for j in range(i + 1, exams):

            if conflicts[i][j] == 1:

                distance = abs(timetable[i] - timetable[j])

                if distance == 0:
                    penalty += 100
                elif distance == 1:
                    penalty += 16
                elif distance == 2:
                    penalty += 8
                elif distance == 3:
                    penalty += 4
                elif distance == 4:
                    penalty += 2
                elif distance == 5:
                    penalty += 1

    return penalty / students


# ============================================================
# GENERATE FEASIBLE TIMETABLE
# ============================================================

def generate_timetable(num_exams, num_slots, conflicts):

    timetable = [-1] * num_exams

    for exam in range(num_exams):

        slots = list(range(num_slots))
        random.shuffle(slots)

        for slot in slots:

            feasible = True

            for other_exam in range(num_exams):

                if timetable[other_exam] == slot:
                    if conflicts[exam][other_exam] == 1:
                        feasible = False
                        break

            if feasible:
                timetable[exam] = slot
                break

        if timetable[exam] == -1:
            timetable[exam] = random.randint(0, num_slots - 1)

    return timetable


# ============================================================
# GENETIC ALGORITHM
# ============================================================

def tournament_selection(population, fitness_scores, k=5):

    selected = random.sample(list(zip(population, fitness_scores)), k)
    selected.sort(key=lambda x: x[1])

    return deepcopy(selected[0][0])


def crossover(parent1, parent2):

    point = random.randint(1, len(parent1) - 2)

    child = parent1[:point] + parent2[point:]

    return child


def mutate(solution, mutation_rate, num_slots):

    for i in range(len(solution)):

        if random.random() < mutation_rate:
            solution[i] = random.randint(0, num_slots - 1)

    return solution


def genetic_algorithm(conflicts,
                      students,
                      num_slots,
                      population_size=120,
                      generations=500,
                      crossover_rate=0.8,
                      mutation_rate=0.1):

    num_exams = len(conflicts)

    population = [
        generate_timetable(num_exams, num_slots, conflicts)
        for _ in range(population_size)
    ]

    best_solution = None
    best_score = float('inf')

    for generation in range(generations):

        fitness_scores = [
            calculate_penalty(individual, conflicts, students)
            for individual in population
        ]

        current_best = min(fitness_scores)

        if current_best < best_score:
            best_score = current_best
            best_solution = deepcopy(population[np.argmin(fitness_scores)])

        new_population = []

        elite_indices = np.argsort(fitness_scores)[:2]

        for idx in elite_indices:
            new_population.append(deepcopy(population[idx]))

        while len(new_population) < population_size:

            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)

            child = mutate(child, mutation_rate, num_slots)

            new_population.append(child)

        population = new_population

    return best_solution, best_score


# ============================================================
# SIMULATED ANNEALING
# ============================================================

def neighbour(solution, num_slots):

    candidate = deepcopy(solution)

    exam = random.randint(0, len(solution) - 1)

    candidate[exam] = random.randint(0, num_slots - 1)

    return candidate


def swap_neighbour(solution):

    candidate = deepcopy(solution)

    i, j = random.sample(range(len(solution)), 2)

    candidate[i], candidate[j] = candidate[j], candidate[i]

    return candidate


def simulated_annealing(conflicts,
                        students,
                        num_slots,
                        initial_temp=1000,
                        cooling_rate=0.995,
                        iterations=5000):

    num_exams = len(conflicts)

    current = generate_timetable(num_exams, num_slots, conflicts)

    current_score = calculate_penalty(current, conflicts, students)

    best = deepcopy(current)
    best_score = current_score

    temperature = initial_temp

    for iteration in range(iterations):

        if random.random() < 0.5:
            candidate = neighbour(current, num_slots)
        else:
            candidate = swap_neighbour(current)

        candidate_score = calculate_penalty(candidate,
                                            conflicts,
                                            students)

        delta = candidate_score - current_score

        if delta < 0:
            current = candidate
            current_score = candidate_score

        else:
            probability = math.exp(-delta / temperature)

            if random.random() < probability:
                current = candidate
                current_score = candidate_score

        if current_score < best_score:
            best = deepcopy(current)
            best_score = current_score

        temperature *= cooling_rate

        if temperature < 0.001:
            break

    return best, best_score


# ============================================================
# PARTICLE SWARM OPTIMISATION
# ============================================================

def particle_swarm_optimisation(conflicts,
                                students,
                                num_slots,
                                swarm_size=50,
                                iterations=300,
                                inertia=0.7,
                                cognitive=1.5,
                                social=1.5):

    num_exams = len(conflicts)

    particles = [
        generate_timetable(num_exams, num_slots, conflicts)
        for _ in range(swarm_size)
    ]

    personal_best = deepcopy(particles)

    personal_scores = [
        calculate_penalty(p, conflicts, students)
        for p in particles
    ]

    global_best = deepcopy(personal_best[np.argmin(personal_scores)])
    global_score = min(personal_scores)

    for iteration in range(iterations):

        for i in range(swarm_size):

            particle = particles[i]

            for exam in range(num_exams):

                r1 = random.random()
                r2 = random.random()

                if r1 < cognitive * 0.1:
                    particle[exam] = personal_best[i][exam]

                if r2 < social * 0.1:
                    particle[exam] = global_best[exam]

                if random.random() < inertia * 0.05:
                    particle[exam] = random.randint(0,
                                                    num_slots - 1)

            score = calculate_penalty(particle,
                                      conflicts,
                                      students)

            if score < personal_scores[i]:
                personal_best[i] = deepcopy(particle)
                personal_scores[i] = score

            if score < global_score:
                global_best = deepcopy(particle)
                global_score = score

    return global_best, global_score


# ============================================================
# EXPERIMENTAL EXECUTION
# ============================================================

def run_experiment(dataset_name, config):

    exams = config['exams']
    students = config['students']
    slots = config['slots']

    print("\n================================================")
    print(f"DATASET: {dataset_name}")
    print("================================================")

    conflicts = generate_conflict_matrix(exams)

    # Genetic Algorithm
    start = time.time()

    _, ga_score = genetic_algorithm(
        conflicts,
        students,
        slots,
        generations=300
    )

    ga_time = round(time.time() - start, 2)

    # Simulated Annealing
    start = time.time()

    _, sa_score = simulated_annealing(
        conflicts,
        students,
        slots,
        iterations=5000
    )

    sa_time = round(time.time() - start, 2)

    # Particle Swarm Optimisation
    start = time.time()

    _, pso_score = particle_swarm_optimisation(
        conflicts,
        students,
        slots,
        iterations=300
    )

    pso_time = round(time.time() - start, 2)

    # Results
    print(f"GA  -> Penalty: {round(ga_score, 2)} | Time: {ga_time}s")
    print(f"SA  -> Penalty: {round(sa_score, 2)} | Time: {sa_time}s")
    print(f"PSO -> Penalty: {round(pso_score, 2)} | Time: {pso_time}s")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    random.seed(42)
    np.random.seed(42)

    for dataset_name, config in DATASETS.items():
        run_experiment(dataset_name, config)


if __name__ == "__main__":
    main()
