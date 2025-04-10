import random


def real_to_binary(x, domain_min, domain_max, num_bits):
    # mapping de la [domain_min, domain_max] la [0, 2^num_bits - 1].
    scale = (x - domain_min) / (domain_max - domain_min)
    int_val = round(scale * (2 ** num_bits - 1))
    bin_str = format(int_val, f'0{num_bits}b')
    return bin_str


def binary_to_real(bin_str, domain_min, domain_max):
    int_val = int(bin_str, 2)
    scale = int_val / (2 ** len(bin_str) - 1)
    x = domain_min + scale * (domain_max - domain_min)
    return x


def calc_num_bits(domain_min, domain_max, precision):
    interval = (domain_max - domain_min) * (10 ** precision)
    num_bits = 0
    while (2 ** num_bits - 1) < interval:
        num_bits += 1
    return num_bits


def fitness_function(x, a, b, c, d, e):
    return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e


def initialize_population(pop_size, num_bits):
    population = []
    for _ in range(pop_size):
        chrom = ''.join(random.choice(['0', '1']) for _ in range(num_bits))
        population.append(chrom)
    return population


def evaluate_population(population, domain_min, domain_max, a, b, c, d, e):
    decoded = [binary_to_real(ch, domain_min, domain_max) for ch in population]
    fitnesses = [fitness_function(x, a, b, c, d, e) for x in decoded]
    return decoded, fitnesses


def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2, point


def detailed_roulette_wheel_selection(population, fitnesses, f, generation):
    f.write(f"\n---- DETAILED SELECTION PROCESS (GEN {generation}) ----\n")

    total_fit = sum(fitnesses)
    if total_fit == 0:
        f.write("All fitnesses are 0! Selecting randomly.\n")
        return random.choices(population, k=len(population))

    p_i = [fi / total_fit for fi in fitnesses]
    q = []
    current = 0.0
    for pi_val in p_i:
        current += pi_val
        q.append(current)

    for idx, (pval, qval) in enumerate(zip(p_i, q)):
        f.write(f"Ind {idx + 1}: p={pval:.6f}, q={qval:.6f}\n")

    new_pop = []
    for sel_i in range(len(population)):
        u = random.random()
        left = 0
        right = len(q) - 1
        chosen_index = 0

        while left <= right:
            mid = (left + right) // 2
            if u <= q[mid]:
                chosen_index = mid
                right = mid - 1
            else:
                left = mid + 1

        new_pop.append(population[chosen_index])

        f.write(f"u={u:.6f} -> interval of Ind {chosen_index + 1}\n")

    return new_pop


def crossover_population(selected_pop, crossover_prob, f, generation):
    mating_pool = []
    new_population = []
    for i, chrom in enumerate(selected_pop):
        if random.random() < crossover_prob:
            mating_pool.append(chrom)
            f.write(f"Ind {i + 1} enters mating pool\n")
        else:
            new_population.append(chrom)
            f.write(f"Ind {i + 1} does NOT enter mating pool\n")

    random.shuffle(mating_pool)
    f.write(f"Mating pool size: {len(mating_pool)}\n")

    for i in range(0, len(mating_pool), 2):
        if i + 1 < len(mating_pool):
            p1 = mating_pool[i]
            p2 = mating_pool[i + 1]
            c1, c2, cut_point = single_point_crossover(p1, p2)

            new_population.append(c1)
            new_population.append(c2)
            f.write(f"  Pair (cut={cut_point}): {p1} x {p2} -> {c1}, {c2}\n")
        else:
            new_population.append(mating_pool[i])
            f.write(f"  Unpaired chrom: {mating_pool[i]}\n")

    return new_population


def mutate_population(population, mutation_prob, f, generation):
    f.write(f"\n---- MUTATION PROCESS (GEN {generation}) ----\n")

    mutated_pop = []
    for idx, chrom in enumerate(population):
        new_chrom = ""
        for bit in chrom:
            if random.random() < mutation_prob:
                new_chrom += '1' if bit == '0' else '0'
            else:
                new_chrom += bit
        mutated_pop.append(new_chrom)

    return mutated_pop


def run_genetic_algorithm(
        pop_size=20,
        domain_min=-2,
        domain_max=2.3,
        a=-1.0,
        b=0.0,
        c=4.0,
        d=2.0,
        e=4.0,
        precision=6 ,
        crossover_prob=0.25,
        mutation_prob=0.01,
        num_generations=50,
        report_file="Evolutie.txt"
):
    num_bits = calc_num_bits(domain_min, domain_max, precision)

    population = initialize_population(pop_size, num_bits)

    with open(report_file, 'w') as f:
        f.write("INPUTS\n")
        f.write("------------------------------------\n")
        f.write(f"Population size: {pop_size}\n")
        f.write(f"Domain: [{domain_min}, {domain_max}]\n")
        f.write(f"Function: f(x) = {a}*x^2 + {b}*x + {c}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Crossover probability: {crossover_prob}\n")
        f.write(f"Mutation probability: {mutation_prob}\n")
        f.write(f"Number of generations: {num_generations}\n")
        f.write(f"Bits per individual: {num_bits}\n\n")

        decoded, fitnesses = evaluate_population(population, domain_min, domain_max, a, b, c, d, e)

        f.write("INITIAL POPULATION (Generation 0)\n")
        for i, (chrom, x_val, fit_val) in enumerate(zip(population, decoded, fitnesses)):
            f.write(f"Ind {i + 1}: Bits={chrom}, X={x_val:.6f}, f(X)={fit_val:.6f}\n")

        for gen in range(1, num_generations + 1):
            best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
            best_individual = population[best_idx]
            best_fitness = fitnesses[best_idx]
            best_x = decoded[best_idx]

            # selectia
            if gen == 1:
                selected_pop = detailed_roulette_wheel_selection(population, fitnesses, f, gen)
            else:
                total_fit = sum(fitnesses)
                if total_fit == 0:
                    selected_pop = random.choices(population, k=len(population))
                else:
                    p_i = [fi / total_fit for fi in fitnesses]
                    q = []
                    cs = 0.0
                    for val in p_i:
                        cs += val
                        q.append(cs)

                    selected_pop = []
                    for _ in range(len(population)):
                        u = random.random()

                        left, right = 0, len(q) - 1
                        chosen = 0
                        while left <= right:
                            mid = (left + right) // 2
                            if u <= q[mid]:
                                chosen = mid
                                right = mid - 1
                            else:
                                left = mid + 1
                        selected_pop.append(population[chosen])

            # crossover
            if gen == 1:
                crossed_pop = crossover_population(selected_pop, crossover_prob, f, gen)
                f.write(f"\nPopulation AFTER CROSSOVER (Gen {gen}):\n")
                for i, chrom in enumerate(crossed_pop):
                    f.write(f"  Ind {i + 1}: {chrom}\n")
            else:
                mating_pool = []
                new_population = []
                for i, chrom in enumerate(selected_pop):
                    if random.random() < crossover_prob:
                        mating_pool.append(chrom)
                    else:
                        new_population.append(chrom)
                random.shuffle(mating_pool)

                for i in range(0, len(mating_pool), 2):
                    if i + 1 < len(mating_pool):
                        p1 = mating_pool[i]
                        p2 = mating_pool[i + 1]
                        c1, c2, cut_point = single_point_crossover(p1, p2)
                        new_population.append(c1)
                        new_population.append(c2)
                    else:
                        new_population.append(mating_pool[i])

            # mutation
            if gen == 1:
                mutated_pop = mutate_population(crossed_pop, mutation_prob, f, gen)
                f.write(f"\nPopulation AFTER MUTATION (Gen {gen}):\n")
                for i, chrom in enumerate(mutated_pop):
                    f.write(f"  Ind {i + 1}: {chrom}\n")
                f.write("\n")
            else:
                mutated_pop = []
                for chrom in crossed_pop:
                    new_chrom = ""
                    for bit in chrom:
                        if random.random() < mutation_prob:
                            new_chrom += '1' if bit == '0' else '0'
                        else:
                            new_chrom += bit
                    mutated_pop.append(new_chrom)

            # elitism, pun cel mai bun in loc de cel mai prost
            decoded_mut, fits_mut = evaluate_population(mutated_pop, domain_min, domain_max, a, b, c, d, e)
            worst_idx = min(range(len(mutated_pop)), key=lambda i: fits_mut[i])
            mutated_pop[worst_idx] = best_individual

            population = mutated_pop

            # evaluez populatia noua
            decoded, fitnesses = evaluate_population(population, domain_min, domain_max, a, b, c, d, e)
            gen_best_fitness = max(fitnesses)
            gen_mean_fitness = sum(fitnesses) / len(fitnesses)
            best_gen_idx = max(range(len(population)), key=lambda i: fitnesses[i])
            best_gen_x = decoded[best_gen_idx]

            # populatia noua
            if gen == 1:
                f.write(f"--- END OF GEN {gen} (DETAILED) ---\n")
                f.write(
                    f"Best X = {best_gen_x:.6f}, Max fitness = {gen_best_fitness:.6f}, Mean fitness = {gen_mean_fitness:.6f}\n\n")
            else:
                f.write(
                    f"Gen {gen}: Best X = {best_gen_x:.6f}, Max fitness = {gen_best_fitness:.6f}, Mean fitness = {gen_mean_fitness:.6f}\n")



run_genetic_algorithm()
