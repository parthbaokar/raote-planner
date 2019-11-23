import random

def clean_cities():
    with open('master_names.txt', 'r') as file:
        name_input = file.readlines()
    name_input = [line.rstrip() for line in name_input]
    name_output = set()
    for full_name in name_input:
        if (len(full_name.split()) >= 2):
            continue
        if full_name.isalnum() and len(full_name) < 15:
            name_output.add(full_name)
    open('master_names.txt', 'w').close()
    with open('master_names.txt', 'w') as file:
        for s in name_output:
            file.write("%s\n" % s)

def choose_cities(number):
    input_file = open('master_names.txt', 'r')

    name_input = input_file.readlines()
    name_input = [line.rstrip() for line in name_input]
    if number < len(name_input):
        choice = random.sample(name_input, k=number)
        return ' '.join(choice)
    else:
        raise ValueError('Must choose a number of cities less than the size of the master list of cities.')

def choose_homes(cities, number):
    if number < len(cities):
        choice = random.sample(cities, k=number)
        return ' '.join(choice)
    else:
        raise ValueError('Must choose a number of cities less than the size of the master list of cities.')
