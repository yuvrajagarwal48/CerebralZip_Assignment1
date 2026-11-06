#!/usr/bin/env python3
"""
================================================================================
Centralized Activity Text Prompts
================================================================================
Defines balanced, visible physical-action textual descriptions for all 10
canonical activity classes for zero-shot text-video contrastive alignment (X-CLIP).
All pipeline and evaluation components must import prompts exclusively from this module.
================================================================================
"""

from typing import Dict, List

ACTIVITY_PROMPTS: Dict[str, List[str]] = {
    "Fighting/Physical Assault": [
        "Two or more people engage in physical aggression or confrontation involving deliberate physical contact. The activity may include punching, kicking, slapping, pushing, hitting, wrestling, grappling, or repeatedly attacking another person.",
        "A person punching another person during a physical fight.",
        "A person kicking another person with their legs.",
        "Two people repeatedly hitting and attacking each other.",
        "Two people wrestling, grappling or struggling physically.",
        "A person pushing another person aggressively to the ground.",
        "A person slapping or striking another person.",
        "One person physically assaulting another person while the other attempts to defend themselves.",
        "Two or more people engaging in a violent physical confrontation.",
        "A person repeatedly striking another individual who is on the ground.",
        "A person chasing and physically attacking another person."
    ],
    "Falling": [
        "A person unexpectedly loses balance, slips, trips, collapses or falls from an upright or elevated position onto the ground or another surface. The activity involves a noticeable downward movement of the person's body.",
        "A person slipping and falling onto the ground.",
        "A person tripping while walking and falling forward.",
        "A person losing balance and falling backward.",
        "A person suddenly collapsing onto the floor.",
        "A person falling down while running.",
        "A person stumbling and falling after losing balance.",
        "A person falling from a chair, staircase or elevated surface.",
        "A person falling onto the ground after being pushed.",
        "A person dropping from a standing position to the ground.",
        "A person struggling to maintain balance before falling."
    ],
    "Fire/Smoke": [
    # General fire and smoke
    "A video showing an active fire or visible smoke in the environment.",
    "A scene containing visible flames, burning materials or thick smoke.",
    "A fire incident occurring in an indoor or outdoor environment.",

    # Visible flames
    "Bright orange and yellow flames burning and flickering in a scene.",
    "Large flames rising from a burning object or structure.",
    "A building engulfed in fire with flames coming out of windows.",
    "Flames spreading across the surface of an object.",
    "A pile of materials actively burning with visible flames.",
    "A fire burning on the ground and spreading to nearby objects.",
    "A vehicle on fire with flames coming from its body or engine.",
    "A trash bin or garbage pile burning with visible flames.",
    "A small fire producing continuous flickering flames.",
    "An intense fire with flames rising high into the air.",

    # Smoke
    "Thick dark smoke rising into the air from a particular location.",
    "Large clouds of black smoke emerging from a burning building.",
    "Dense grey smoke billowing from a vehicle or other object.",
    "A continuous plume of smoke rising from a fire.",
    "Smoke spreading through an indoor room or enclosed space.",
    "A large area covered by thick smoke from a fire incident.",
    "White smoke rising from an object that is burning.",
    "Black smoke coming out of windows, doors or openings of a building.",
    "A distant fire producing a large visible smoke column.",

    # Fire spreading and developing
    "A fire starting and gradually producing visible flames.",
    "Flames growing larger and spreading across nearby materials.",
    "A burning object producing increasingly thick smoke.",
    "An area becoming engulfed in flames and smoke.",
    "Fire spreading from one object to another.",

    # Small or partially visible fires
    "A small flame burning on an object in an otherwise normal environment.",
    "A small fire burning near a wall or building.",
    "Visible flames emerging from behind an object.",
    "A partially obscured fire producing visible smoke.",
    "A distant burning object with visible flames or smoke."
],
    "Intrusion/Trespassing": [
        "A person enters, attempts to enter or moves into an area that appears restricted, enclosed or protected without an obvious authorized entry process. Visible actions may include climbing barriers, bypassing gates, forcing doors, entering through windows or accessing restricted premises.",
        "A person climbing over a wall or fence to enter an enclosed property.",
        "A person entering a building through a window.",
        "A person forcing open a closed or locked door.",
        "A person attempting to enter a restricted area through a gate.",
        "A person bypassing a security barrier to access a property.",
        "A person entering an enclosed area without using its designated entrance.",
        "A person breaking into a building through an alternative entry point.",
        "A person attempting to open a secured entrance.",
        "A person entering a restricted compound by climbing a boundary wall.",
        "A person moving into a clearly marked restricted-access area."
    ],
    "Loitering": [
        "A person remains in the same general location for an unusually extended period without apparent progression toward a destination or clearly visible activity. The person may repeatedly stand, sit, wait, wander around a small area or move back and forth.",
        "A person standing in one location for an extended period.",
        "A person repeatedly pacing around the same small area.",
        "A person sitting in an area for a prolonged duration.",
        "A person wandering around a building entrance without leaving.",
        "A person repeatedly approaching and moving away from the same location.",
        "A person standing near a doorway for an unusually long time.",
        "A person lingering around a vehicle or property.",
        "A person walking back and forth in a confined area.",
        "A person remaining near an entrance without entering or leaving.",
        "A person repeatedly stopping and waiting in the same location."
    ],
    "Vandalism": [
        "A person deliberately damages, destroys, defaces or interferes with public or private property. Visible actions may include smashing, breaking, kicking, hitting, spraying, scratching, overturning or damaging objects and structures.",
        "A person smashing a window with an object.",
        "A person kicking and damaging a door or other property.",
        "A person breaking objects by hitting them repeatedly.",
        "A person spray-painting graffiti on a wall.",
        "A person scratching or defacing a public surface.",
        "A person deliberately overturning bins, furniture or other objects.",
        "A person damaging a parked vehicle.",
        "A person throwing objects at windows or buildings.",
        "A person breaking street furniture or public infrastructure.",
        "A person repeatedly striking an object until it breaks."
    ],
    "Theft/Chain Snatching": [
        "A person takes, attempts to take or forcibly removes an object or personal belonging that appears to belong to another person. The activity may involve secretly picking up unattended belongings, grabbing objects from someone, snatching bags or accessories, or taking items and quickly leaving the scene.",
        "A person stealing a bag belonging to another person.",
        "A person forcibly snatching an object from someone's hand.",
        "A person grabbing a handbag and running away.",
        "A person taking a mobile phone from another person's pocket or hand.",
        "A person stealing belongings from a table or unattended location.",
        "A person removing an object from another person's possession.",
        "A person grabbing a necklace or chain from someone's neck.",
        "A person taking an object and quickly fleeing the scene.",
        "A person reaching into another person's bag and removing an item.",
        "A person stealing an object from a parked vehicle."
    ],
    "Crowd Formation/Unusual Gathering": [
        "Multiple people gather or converge in the same general location, forming a noticeable group or crowd. The activity may involve people arriving from different directions, clustering around a particular location, remaining together or rapidly increasing in number.",
        "A group of people gathering in an open area.",
        "Multiple people converging toward the same location.",
        "A crowd gradually forming near a building entrance.",
        "Several people gathering around a particular object or incident.",
        "A large group of people clustering together in a public area.",
        "Multiple individuals approaching and joining an existing group.",
        "People forming a dense gathering around a particular location.",
        "A group of people suddenly assembling in the street.",
        "A crowd growing as more people arrive.",
        "Several people surrounding another person or an object."
    ],
    "Running in a Restricted Zone": [
        "A person moves rapidly through an area by running or sprinting rather than walking. The activity involves clear running motion, such as alternating strides, rapid leg movement and forward displacement, within an area where running is not permitted.",
        "A person sprinting through a designated restricted area.",
        "A person running rapidly through a building corridor.",
        "A person running inside an area marked as a no-running zone.",
        "A person sprinting through a restricted public facility.",
        "A person running through a designated pedestrian-only area.",
        "A person rapidly running across an indoor facility.",
        "A person running through a clearly marked prohibited area.",
        "A person running inside a facility where walking is required."
    ],
    "Abandoned Object": [
        "A person places, drops or leaves an object in a location and subsequently moves away without retrieving it. The object remains unattended for a noticeable duration after the person has departed.",
        "A person leaving a bag on the ground and walking away.",
        "A person placing a suitcase in a public area and departing.",
        "A person leaving a package near a building entrance.",
        "A person dropping an object and walking away without retrieving it.",
        "A person placing a backpack on a bench and leaving the area.",
        "A person leaving a box unattended in a corridor.",
        "A person placing an object in a public space and moving out of view.",
        "A person leaving belongings behind while walking away.",
        "An object remaining unattended after the person who brought it has departed.",
        "A person leaving a bag near a crowded area and walking away."
    ]
}

