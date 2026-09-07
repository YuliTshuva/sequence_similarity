import json
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tune_weights import load_sequence

rcParams["font.family"] = "Times New Roman"

names_dict = {
    "Amy": ["April", "Lisa", "Crystal", "Janet", "Sharon", "Brenda", "Lillian", "Margaret", "Alice"],
    "Mary": ["Beverly", "Martha", "Marilyn", "Debra", "Lisa", "Michelle", "Natalie", "Emily", "Sophia"],
    "Denise": ["Susan", "Tammy", "Pamela", "Janet", "Jamie", "Linda", "Olivia", "Alice", "Martha"],
    "Elizabeth": ["Catherine", "Maria", "Beverly", "Crystal", "Amy", "Samantha", "Ashley", "Chloe", "Isabella"],
    "Lillian": ["Helen", "Anna", "Margaret", "Tracy", "Tina", "Laura", "Sydney", "Isabella", "Chloe"],
    "Shirley": ["Linda", "Dolores", "Cindy", "Dorothy", "Christina", "Megan", "Natalie", "Olivia", "Abigail"],
    "Jennifer": ["Christina", "Rachel", "Julie", "Betty", "Linda", "Dorothy", "Mary", "Grace", "Elizabeth"],
    "Lisa": ["Amy", "Kim", "Teresa", "Brenda", "Christina", "Rebecca", "Grace", "Elizabeth", "Emma"],
    "Janet": ["Barbara", "Patricia", "Karen", "Gloria", "Joan", "Virginia", "Emma", "Sophia", "Abigail"],
    "Charlotte": ["Catherine", "Peggy", "Gloria", "Lauren", "Linda", "Debbie", "Alexandra", "Shelby", "Chloe"],
    "Amanda": ["Alexandra", "Kayla", "Hannah", "Joyce", "Rebecca", "Andrea", "Elizabeth", "Alice", "Martha"],
    "Catherine": ["Elizabeth", "Rebecca", "Maria", "Mary", "Grace", "Anna", "Emily", "Emma", "Abigail"],
    "Grace": ["Margaret", "Anna", "Mary", "Julie", "Andrea", "Natalie", "Jessica", "Samantha", "Amanda"],
    "Debbie": ["Debra", "Kathleen", "Deborah", "Donna", "Karen", "Samantha", "Abigail", "Anna", "Elizabeth"],
    "Katherine": ["Elizabeth", "Peggy", "Theresa", "Erin", "Tina", "Sara", "Anna", "Kayla", "Amanda"],
    "Judith": ["Diane", "Kathleen", "Susan", "Peggy", "Beverly", "Marilyn", "Emily", "Natalie", "Anna"],
    "Jacqueline": ["Theresa", "Shirley", "Carolyn", "Erin", "Crystal", "Dorothy", "Sydney", "Mildred", "Martha"],
    "Laura": ["Sharon", "Kelly", "Shannon", "Joyce", "Joan", "Linda", "Grace", "Marie", "Frances"],
    "Cynthia": ["Lisa", "Tina", "Lori", "Christina", "Janet", "Sharon", "Margaret", "Anna", "Grace"],
    "Natalie": ["Emily", "Sarah", "Chloe", "Irene", "Theresa", "Catherine", "Margaret", "Mary", "Marie"],
    "Florence": ["Helen", "Betty", "Cindy", "Christine", "Laura", "Ava", "Natalie", "Victoria", "Emma"],
    "Marilyn": ["Brenda", "Sharon", "Patricia", "Debbie", "Melissa", "Jamie", "Jasmine", "Alexandra", "Anna"],
    "Rebecca": ["Brenda", "Robin", "Paula", "Linda", "Sydney", "Judith", "Lillian", "Helen", "Alice"],
    "Doris": ["Nancy", "Patricia", "Carol", "Sharon", "Shirley", "Catherine", "Andrea", "Anna", "Victoria"],
    "Tracy": ["Amy", "Lisa", "Cynthia", "Brenda", "Janet", "Christine", "Elizabeth", "Lillian", "Anna"],
}

results = {
    "Rebecca": {"Brenda": 1, "Paula": 2, "Robin": 3, "Judith": 4, "Helen": 5, "Sydney": 5, "Irene": 5, "Lillian": 5},
    "Mary": {"Beverly": 1, "Martha": 2, "Marilyn": 3, "Natalie": 4, "Emily": 5, "Michelle": 5},
    "Jacqueline": {"Theresa": 1, "Carolyn": 2, "Shirley": 3, "Martha": 4, "Crystal": 5, "Erin": 5, "Dorothy": 7},
    "Natalie": {"Emily": 1, "Chloe": 1, "Sarah": 1, "Theresa": 4, "Irene": 5, "Marie": 5},
    "Jennifer": {"Julie": 1, "Rachel": 2, "Christina": 2, "Grace": 4, "Betty": 4},
    "Amy": {"Crystal": 1, "April": 2, "Lisa": 3, "Brenda": 4, "Janet": 4, "Alice": 6, "Lillian": 6},
    "Florence": {"Helen": 1, "Cindy": 2, "Betty": 3, "Natalie": 4, "Ava": 4, "Christine": 4},
    "Elizabeth": {"Catherine": 1, "Maria": 2, "Beverly": 2, "Isabella": 4},
    "Katherine": {"Elizabeth": 1, "Theresa": 1, "Peggy": 3, "Erin": 4, "Chloe": 5, "Kayla": 5, "Amanda": 5},
    "Charlotte": {"Shelby": 1, "Peggy": 2, "Catherine": 3, "Gloria": 4, "Chloe": 5, "Lauren": 6, "Alexandra": 7,
                  "Linda": 8},
    "Janet": {"Patricia": 1, "Barbara": 2, "Karen": 2, "Virginia": 4, "Gloria": 5, "Joan": 5},
    "Amanda": {"Hannah": 1, "Kayla": 1, "Alexandra": 3, "Joyce": 4, "Martha": 5, "Rebecca": 5},
    "Catherine": {"Rebecca": 1, "Mary": 2, "Elizabeth": 3, "Anna": 4, "Maria": 5, "Grace": 6, "Emily": 7},
    "Grace": {"Anna": 1, "Margaret": 2, "Mary": 3, "Samantha": 3, "Jessica": 5, "Andrea": 6, "Amanda": 7, "Natalie": 7},
    "Shirley": {"Dolores": 1, "Cindy": 1, "Linda": 3, "Dorothy": 4, "Megan": 5, "Olivia": 6, "Christina": 6},
    "Lillian": {"Anna": 1, "Helen": 2, "Margaret": 3, "Laura": 4, "Tina": 5, "Chloe": 6, "Sydney": 6, "Isabella": 6},
    "Lisa": {"Teresa": 1, "Amy": 2, "Kim": 3, "Brenda": 4, "Christina": 5, "Emma": 6, "Grace": 6, "Rebecca": 6,
             "Elizabeth": 6},
    "Marilyn": {"Brenda": 1, "Sharon": 2, "Patricia": 3, "Jamie": 4, "Melissa": 5, "Alexandra": 6},
    "Denise": {"Tammy": 1, "Susan": 1, "Pamela": 1, "Janet": 4, "Linda": 5, "Martha": 6},
    "Laura": {"Kelly": 1, "Shannon": 2, "Sharon": 3, "Joyce": 4, "Frances": 5, "Joan": 6, "Grace": 6},
    "Doris": {"Carol": 1, "Nancy": 2, "Patricia": 3, "Shirley": 4, "Sharon": 5, "Victoria": 6},
    "Tracy": {"Cynthia": 1, "Amy": 2, "Lisa": 3, "Elizabeth": 4, "Janet": 5},
    "Cynthia": {"Lisa": 1, "Lori": 1, "Tina": 1, "Anna": 4, "Janet": 4, "Grace": 6, "Sharon": 6},
    "Debbie": {"Deborah": 1, "Debra": 2, "Kathleen": 3, "Karen": 4, "Elizabeth": 5},
    "Judith": {"Diane": 1, "Kathleen": 2, "Susan": 2, "Beverly": 4, "Peggy": 4, "Emily": 6}
}

dataset = {}

for name, value in results.items():
    top_3 = [n for n in value if value[n] <= 3]
    negs = [n for n in names_dict[name] if n not in top_3]
    if len(top_3) + len(negs) != 9:
        print(f"Error for {name}: top_3={top_3}, negs={negs}")
    dataset[name] = {
        "positives": top_3,
        "negatives": negs
    }

    if len(top_3) != 3:
        print(f"Error for {name}: top_3={top_3}")
    if len(negs) != 6:
        print(f"Error for {name}: negs={negs}")

# # Save the dataset to a JSON file
# with open(join("data", "dataset.json"), "w") as f:
#     json.dump(dataset, f, indent=4)

# Plot an example for anchor and the top 3 positives and 6 negatives
anchor_name = "Doris"
anchor_positives = dataset[anchor_name]["positives"]
anchor_negatives = dataset[anchor_name]["negatives"]

# Plotting
plt.subplots(4, 3, figsize=(15, 20))

# Plot the anchor
plt.subplot(4, 3, 2)
plt.title(f"Anchor: {anchor_name}")
seq = load_sequence(anchor_name)
plt.plot(range(len(seq)), seq, color="blue")
plt.ylim(0, 1)

# Plot positives
for i, pos in enumerate(anchor_positives):
    plt.subplot(4, 3, i + 4)
    plt.title(f"Positive {i + 1}: {pos}")
    seq = load_sequence(pos)
    plt.plot(range(len(seq)), seq, color="turquoise")
    plt.ylim(0, 1)

# Plot negatives
for i, neg in enumerate(anchor_negatives):
    plt.subplot(4, 3, i + 7)
    plt.title(f"Negative {i + 1}: {neg}")
    seq = load_sequence(neg)
    plt.plot(range(len(seq)), seq, color="salmon")
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()