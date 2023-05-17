# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List

from opentrons import protocol_api
import itertools
import math
import random


class Well:
    ''' Well that knows its location and whether it can be transferred from '''
    location: str
    current_volume: float
    maximum_volume: float

    # https://www.scaler.com/topics/multiple-constructors-python/
    def __init__(self, *args):
        if len(args) == 0:
            raise Exception("must give at least location of the Well")
        elif len(args) == 1:
            self.location = args[0]
            self.current_volume = 0
            self.maximum_volume = 0
        elif len(args) == 2:
            self.location = args[0]
            self.current_volume = args[1]
            self.maximum_volume = args[1] # set same as current
        elif len(args) == 3:
            self.location = args[0]
            self.current_volume = args[1]
            self.maximum_volume = args[2]
        else:
            raise Exception("too many arguments to Well constructor")

    def can_be_transferred_from(self, volume_to_pipette: float):
        if volume_to_pipette < self.current_volume + SOURCE_WELL_MARGIN:
            return True
        return False


class Species:
    ''' What we are pipetting. Knows from which source
     well to transfer next '''
    name: str
    source_wells: list

    def __init__(self, name: str, source_wells: list):
        self.name = name
        self.source_wells = source_wells

    def get_current_source_well(self, volume_to_transfer: float):
        ''' get the first source well with enough fluid, throws exception if no well with enough fluid is found '''
        for well in self.source_wells:
            well: Well
            if well.can_be_transferred_from(volume_to_transfer):
                return well
        raise Exception("Could not find source well with enough fluid for Species", self.name, ". Should more source wells be added? Number of source wells", len(self.source_wells), "amount of fluid in last well:", self.source_wells[-1])


class Combination:
    ''' A combination of Species objects
    that is pipetted into a single target well '''
    specie: list
    target_well: Well

    def __init__(self, specie: list, target_well: Well):
        self.specie = specie
        self.target_well = target_well

    def pipette_combination(self, pipette: protocol_api.InstrumentContext, source_plate: protocol_api.Labware, target_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool):
        ''' Calls the Opentrons transfer method to pipette the Species contained in this Combination '''
        assert len(self.specie) > 0, "specie of combination not initialized"
        spec: Species
        for spec in self.specie:
            assert spec.name is not None and spec.source_wells is not None

        for spec in self.specie:
            volume_to_transfer = total_volume / len(self.specie)
            source_well: Well
            source_well = spec.get_current_source_well(volume_to_transfer)
            initial_volume = source_well.current_volume
            pipette.transfer(volume_to_transfer, source_plate[source_well.location], target_plate[self.target_well.location], change_pipettes)
            source_well.current_volume -= volume_to_transfer
            assert source_well.current_volume == initial_volume - volume_to_transfer, "Source well volume should decrease correctly"


##########################
# This part can be modified by the user:
# script starting values
SOURCE_WELLS_INITIAL_VOLUME_UL = 10000.0  # modify this based on initial volume; affects how source well is changed
SOURCE_WELL_MARGIN = 10.0  # how many ul is left to source wells before changing to another
# 1: name, 2: list of source wells (incl well location and fluid volume)
SPECIES_1 = Species("laji1", [Well("A1", SOURCE_WELLS_INITIAL_VOLUME_UL)])
SPECIES_2 = Species("laji2", [Well("B1", SOURCE_WELLS_INITIAL_VOLUME_UL)])
SPECIES_3 = Species("laji3", [Well("C1", SOURCE_WELLS_INITIAL_VOLUME_UL)])
SPECIES_4 = Species("laji4", [Well("D1", SOURCE_WELLS_INITIAL_VOLUME_UL)])
SPECIES_5 = Species("laji5", [Well("E1", SOURCE_WELLS_INITIAL_VOLUME_UL)])
SPECIES_6 = Species("laji6", [Well("F1", SOURCE_WELLS_INITIAL_VOLUME_UL)])
SPECIES_LIST = [SPECIES_1, SPECIES_2, SPECIES_3, SPECIES_4, SPECIES_5, SPECIES_6]
NUMBER_OF_SPECIES = len(SPECIES_LIST)  # trivial but maybe useful to see
VOLUME_PER_TARGET_WELL_UL = 60.0
'''
 "kontaminaation ehkäisemiseksi robotin voi määrätä
 koskemaan pipetin kärjellä kaivoin reunaa kun se on pipetoinut nesteen kaivoon"
'''
TOUCH_SIDE_AFTER_DISPENSE = True  # TODO not used yet
####################


def flatten_list(l: list):
    ''' Transform to a single-leveled list
    https://stackoverflow.com/a/952952 '''
    return [item for sublist in l for item in sublist]


def initialize_combinations_list(species_list, source_plate: protocol_api.Labware):
    number_of_species = len(species_list)

    # TODO use source_plate, refactor better

    ''' Initialize combinations list with all the possible
     binomial coefficient combinations of the SPECIES_LIST
      into a randomized single-level list '''
    combinations_list = []

    # https://stackoverflow.com/a/464882
    # https://docs.python.org/3/library/itertools.html#itertools.combinations
    for i in range(0, number_of_species):
        # print("appending", list(itertools.combinations(SPECIES_LIST, i + 1)))
        combinations_list.append(list(itertools.combinations(species_list, i + 1)))

    combinations_list = flatten_list(combinations_list)
    combinations_list: list

    # test that we have the correct number of combinations (63 combinations with 6 species)
    assert len(combinations_list) == sum(math.comb(number_of_species, x) for x in range(1, number_of_species+1)), "combinations_list length should be equal to amount of combinations"
    random.shuffle(combinations_list)  # randomize the order

    collections_list = []

    coords = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]*12  # TODO for testing; do this better, use Opentrons API

    for i, comb in enumerate(combinations_list):
        collections_list.append(Combination(comb, Well(coords[i])))

    # collections_list = list(map(lambda combs: Combination(combs, Well("A1")), combinations_list))

    return collections_list


# https://docs.opentrons.com/v2/new_examples.html#plate-mapping
test_volumes = [
        1,  2,  3,  4,  5,  6,  7,  8,
        9,  10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64,
        65, 66, 67, 68, 69, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96
      ]

assert len(SPECIES_LIST) == NUMBER_OF_SPECIES, "number of species and species list length should be the same"

# https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
metadata = {"apiLevel": "2.13"}  # ylin taso jolla opentrons_simulate toimii


def run(protocol: protocol_api.ProtocolContext):
    # https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
    tiprack_20ul_1 = protocol.load_labware("opentrons_96_filtertiprack_20ul", 1) # saatetaan tarvita monta
    tiprack_20ul_2 = protocol.load_labware("opentrons_96_filtertiprack_20ul", 2)
    tiprack_20ul_3 = protocol.load_labware("opentrons_96_filtertiprack_20ul", 3)
    tiprack_300ul_1 = protocol.load_labware("opentrons_96_tiprack_300ul", 4)  # we need A LOT of tips ...
    tiprack_300ul_2 = protocol.load_labware("opentrons_96_tiprack_300ul", 5)
    tiprack_300ul_3 = protocol.load_labware("opentrons_96_tiprack_300ul", 6)
    tiprack_300ul_4 = protocol.load_labware("opentrons_96_tiprack_300ul", 7)
    tiprack_300ul_5 = protocol.load_labware("opentrons_96_tiprack_300ul", 8)

    master_plate1 = protocol.load_labware("biorad_96_wellplate_200ul_pcr", 9)  # pitää ilmeisesti muuttaa Sarstedtin custom-levyksi
    master_plate2 = protocol.load_labware("biorad_96_wellplate_200ul_pcr", 10)
    reservoir_plate = protocol.load_labware("biorad_96_wellplate_200ul_pcr", 11)

    global_combinations_list: List[Combination]
    global_combinations_list = initialize_combinations_list(SPECIES_LIST, reservoir_plate)

    # https://docs.opentrons.com/v2/new_pipette.html#pipette-models
    left = protocol.load_instrument("p20_single_gen2", "left", [tiprack_20ul_1, tiprack_20ul_2, tiprack_20ul_3])
    right = protocol.load_instrument("p300_single_gen2", "right", [tiprack_300ul_1, tiprack_300ul_2, tiprack_300ul_3, tiprack_300ul_4, tiprack_300ul_5])

    # https://docs.opentrons.com/v2/writing.html#commands
    # https://docs.opentrons.com/v2/new_examples.html
    # left.transfer(15, reservoir_plate['A1'], master_plate1['A1'])
    # left.transfer(test_volumes[0:12] * 8, [reservoir_plate['A1'], reservoir_plate['A3'], reservoir_plate['A5']], master_plate1.wells())
    # right.transfer(test_volumes[::-1], reservoir_plate.wells(), master_plate2.wells())

    # transfer all the combinations
    for combination in global_combinations_list:
        combination.pipette_combination(right, reservoir_plate, master_plate1, 60.0, False)




