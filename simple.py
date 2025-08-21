import re
import numpy as np
import pandas as pd
from rich import print as rprint
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm 

from rcode import *
from matplotlib import pyplot as plt

import torch
import transformer_lens
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

torch.set_grad_enabled(False)
from transformers import GenerationConfig

# --
# Helpers

def to_np(x):
    return x.cpu().numpy()

# Strip repeated eos_token from end of string
def strip_repeated_eos(text, eos_token):
    escaped_eos = re.escape(eos_token)
    pattern     = f"({escaped_eos})+$"
    return re.sub(pattern, eos_token, text)

def extract_choice(x):
    pattern = r'<choice>(.*?)</choice>'
    matches = re.findall(pattern, x)
    if len(matches) == 0:
        return '<error>'
    
    return matches[-1] if matches else None

def n_tokens(x):
    # does this handle special tokens correctly?
    return len(model.tokenizer.encode(x))

def remove_suffix(x, suffix):
    if x.endswith(suffix):
        return x[:-len(suffix)]
    return x

# --

model_str = "Qwen/Qwen2-7B-Instruct"
generation_config = GenerationConfig.from_pretrained(model_str)

model = transformer_lens.HookedTransformer.from_pretrained(model_str)

model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token    = model.tokenizer.eos_token

# --
# Generation

model.reset_hooks()

facts = [
    "A group of owls is called a parliament.",
    "Butterflies taste with their feet.",
    "Sloths only defecate once a week and can lose up to a third of their body weight doing so.",
    "A snail can sleep for three years.",
    "The heart of a shrimp is in its head.",
    "Octopuses have three hearts: two pump blood through the gills, and one circulates it to the rest of the body.",
    "A chameleon's tongue is twice as long as its body.",
    "Giraffes have the same number of neck vertebrae as humans: seven.",
    "The only mammal capable of true sustained flight is the bat.",
    "Sharks can detect a single drop of blood in an Olympic-sized swimming pool.",
    "Hummingbirds are the only birds that can fly backward.",
    "The tusk of a narwhal is an elongated canine tooth, which can grow up to 10 feet long.",
    "Penguins are birds, but they cannot fly; they are excellent swimmers.",
    "Kangaroos use their tails for balance and as a fifth limb.",
    "Dolphins sleep with one eye open and half their brain awake.",
    "A group of rhinoceroses is called a crash.",
    "Starfish do not have brains.",
    "Panda bears eat up to 12-38 kilograms of bamboo per day.",
    "Elephants are the only animals that cannot jump.",
    "Honeybees can flap their wings 200 times per second.",
    "A tiger's stripes are unique to each individual, much like human fingerprints.",
    "Male seahorses carry the eggs and give birth.",
    "Frogs drink water through their skin.",
    "An ostrich's eye is larger than its brain.",
    "Polar bears have black skin under their white fur to absorb sunlight and stay warm.",
    "The platypus is one of only two mammals that lay eggs (the other being the echidna).",
    "Ferrets sleep up to 75% of the day.",
    "A group of porcupines is called a prickle.",
    "The peregrine falcon is the fastest bird in the world, reaching speeds over 200 mph in a dive.",
    "Barnacles have the largest penis-to-body size ratio of any animal.",
    "Some species of jellyfish are biologically immortal.",
    "The blue whale is the largest animal on Earth, weighing as much as 30 elephants.",
    "Rats can go longer without water than camels.",
    "The average housefly lives for about 15 to 30 days.",
    "Cheetahs can run up to 70 miles per hour, but only for short bursts.",
    "Cats can make over 100 different sounds, while dogs can only make about 10.",
    "The longest recorded lifespan of a snail was 14 years.",
    "Crocodiles cannot stick their tongues out.",
    "A group of larks is called an exaltation.",
    "Koala fingerprints are so similar to humans that they have been confused at crime scenes.",
    "Wombats produce cube-shaped poop.",
    "The colossal squid has the largest eyes of any animal.",
    "Sea otters hold hands while they sleep to prevent drifting apart.",
    "Female lions do 90% of the hunting for their pride.",
    "A newborn kangaroo is about the size of a jelly bean.",
    "Snakes smell with their tongues.",
    "Axolotls can regenerate most of their body parts, including limbs, heart, and brain.",
    "Pistol shrimp can create a cavitation bubble that reaches temperatures of 4,500°C.",
    "Some fish can change their sex during their lifetime.",
    "Vampire bats are the only mammals that feed entirely on blood.",
    "The largest known animal migration is that of the Christmas Island red crabs.",
    "Dragonflies have six legs, but they cannot walk.",
    "Glass frogs have translucent skin, allowing you to see their internal organs.",
    "An armadillo's shell is so hard that it can deflect a bullet.",
    "The Gila monster is one of the only venomous lizards in North America.",
    "Woodpeckers use their long tongues to extract insects from trees.",
    "The largest spider in the world is the Goliath birdeater tarantula.",
    "A group of ferrets is called a business.",
    "African elephants have larger ears than Asian elephants, shaped like the continent of Africa.",
    "Most birds do not have a sense of smell.",
    "The average lifespan of a common house mouse is about 1 year.",
    "Zebras are black with white stripes, not white with black stripes.",
    "Geckos can detach their tails to escape predators, and the tail continues to wiggle.",
    "The chameleon's eyes can move independently of each other.",
    "Bears are plantigrade, meaning they walk on the soles of their feet like humans.",
    "Pigeons can recognize themselves in a mirror.",
    "The bombardier beetle can spray a boiling hot, noxious chemical from its abdomen.",
    "Some turtles can breathe through their butts (cloacal respiration).",
    "A group of baboons is called a troop.",
    "The pangolin is the only mammal covered entirely in scales.",
    "Only female mosquitos don't bite; females need blood for egg production.",
    "The fennec fox has the largest ears relative to body size of any canid.",
    "Ostriches lay the largest eggs of any land animal.",
    "A queen bee can lay up to 2,500 eggs per day.",
    "The smallest mammal in the world is the bumblebee bat.",
    "Cows have four stomachs.",
    "Dogs can be trained to detect cancer and other diseases in humans.",
    "Cats cannot taste sweetness.",
    "Scorpions glow under ultraviolet light.",
    "The immortal jellyfish (Turritopsis dohrnii) can revert to its juvenile form after reaching sexual maturity.",
    "Great white sharks can go without eating for three months.",
    "Spiders are not insects; they are arachnids.",
    "Rattlesnakes are born with their rattles, but they don't make a sound until they shed their skin for the first time.",
    "Koalas sleep up to 20 hours a day.",
    "The loudest animal on Earth is the sperm whale, whose clicks can reach 230 decibels.",
    "Emperor penguins can dive deeper than any other bird.",
    "Capybaras are the largest rodents in the world.",
    "A group of crocodiles is called a bask.",
    "Walruses use their whiskers, called vibrissae, to search for food on the seabed.",
    "Dung beetles are the strongest insects, able to pull over 1,000 times their own body weight.",
    "Flamingos are born white; their pink color comes from their diet of brine shrimp and algae.",
    "Sea cucumbers can expel their internal organs as a defense mechanism and then regenerate them.",
    "The Archerfish can shoot down insects from over 6 feet away by spitting water.",
    "Some species of male anglerfish permanently fuse with the female for reproduction.",
    "The smallest fish in the world is Paedocypris progenetica, found in Sumatra.",
    "A chimpanzee can learn to recognize itself in a mirror and use sign language.",
    "The slowest mammal is the three-toed sloth.",
    "Reindeer eyes change color with the seasons to adapt to varying light levels.",
    "A group of squids is called a squad.",
    "The average house cat can jump up to six times its height.",
    "The Komodo dragon has a venomous bite, not just bacteria in its mouth as previously thought.",
    "Beluga whales are known as the 'canaries of the sea' due to their wide range of vocalizations.",
    "Tardigrades, or water bears, are incredibly resilient and can survive in extreme conditions.",
    "A male lion's roar can be heard up to 5 miles away.",
    "Only female silkworms produce silk.",
    "A 'leap' is a group of leopards."
]

input0 = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about an animal. Then output one of "heads" or "tails" wrapped in <choice></choice> tags.'}
]
input_str0 = model.tokenizer.apply_chat_template(input0, tokenize=False, add_generation_prompt=True)
output0    = [input_str0 + f + ' <choice>' for f in facts]

input1 = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about an animal. Then output one of "heads" or "tails" wrapped in <choice></choice> tags.'}
]
input_str1 = model.tokenizer.apply_chat_template(input1, tokenize=False, add_generation_prompt=True)
output1    = [input_str1 + f + '\n\n\n' +  '<choice>' for f in facts]



token_idx  = model.tokenizer.encode('heads')[0]

logits0, _    = model.run_with_cache(output0, padding_side="left", names_filter=lambda hook_name: 'resid_post' in hook_name)
Y0            = logits0[:,-1].softmax(axis=-1)[:,token_idx].to('cpu')

logits1, _    = model.run_with_cache(output1, padding_side="left", names_filter=lambda hook_name: 'resid_post' in hook_name)
Y1            = logits1[:,-1].softmax(axis=-1)[:,token_idx].to('cpu')

_ = plt.scatter(Y0, Y1)
_ = plt.grid('both', alpha=0.25)
show_plot()

from scipy.stats import spearmanr
spearmanr(Y0, Y1)
