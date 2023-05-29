# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List, Tuple

from opentrons import protocol_api
import itertools
import math
import random

from opentrons.protocol_api.labware import OutOfTipsError


class Well:
    """ Well that knows its location and whether it can be transferred from """
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
            self.maximum_volume = args[1]  # set same as current
        elif len(args) == 3:
            if args[1] > args[2]:
                raise Exception(self, "Current volume cannot be higher than maximum volume")
            self.location = args[0]
            self.current_volume = args[1]
            self.maximum_volume = args[2]
        else:
            raise Exception("too many arguments to Well constructor")

    def can_be_transferred_from(self, volume_to_pipette: float):
        if volume_to_pipette < self.current_volume + SOURCE_WELL_MARGIN:
            return True
        return False

    def get_opentrons_well(self, opentrons_plate: protocol_api.Labware):
        return opentrons_plate[self.location]


class Species:
    """ What we are transferring. Knows from which source
     well to transfer next """
    name: str
    source_wells: list

    def __init__(self, name: str, source_wells: list):
        self.name = name
        self.source_wells = source_wells

    def get_current_source_well(self, volume_to_transfer: float):
        """ get the first source well with enough fluid, throws exception if no well with enough fluid is found """
        for well in self.source_wells:
            well: Well
            if well.can_be_transferred_from(volume_to_transfer):
                return well
        raise Exception("Could not find source well with enough fluid for Species", self.name, ". Should more source wells be added? Number of source wells", len(self.source_wells), "amount of fluid in last well:", self.source_wells[-1])


class Combination:
    """ A combination of Species objects
    that is transferred into a single target well """
    specie: list
    target_well: Well

    def __init__(self, specie: list, target_well: Well):
        self.specie = specie
        self.target_well = target_well

    def get_individual_transfer_volume(self, target_well_volume_ul: float):
        return target_well_volume_ul / (len(self.specie)*1.0)

    def get_amount_of_transfers(self):
        return len(self.specie)

    def get_required_tips(self, target_well_volume_ul: float) -> Tuple[int, float]:
        """ Get the amount of tips needed and tip volume """
        tip_volume = define_tip_volume(self.get_individual_transfer_volume(target_well_volume_ul))
        return self.get_amount_of_transfers(), tip_volume

    def pipette_combination(self, pipette: protocol_api.InstrumentContext, source_plate: protocol_api.Labware, target_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool):
        """ Calls the Opentrons transfer method to pipette the Species contained in this Combination """
        assert len(self.specie) > 0, "specie of combination not initialized"
        spec: Species
        for spec in self.specie:
            assert spec.name is not None and spec.source_wells is not None

        for spec in self.specie:
            volume_to_transfer = total_volume / len(self.specie)
            source_well: Well
            source_well = spec.get_current_source_well(volume_to_transfer)
            initial_volume = source_well.current_volume
            pipette.transfer(volume_to_transfer, source_well.get_opentrons_well(source_plate), self.target_well.get_opentrons_well(target_plate), change_pipettes)
            source_well.current_volume -= volume_to_transfer
            assert source_well.current_volume == initial_volume - volume_to_transfer, "Source well volume should decrease correctly"


def define_tip_volume(volume: float) -> float:
    """ Get the right pipette volume for this actual transfer volume """
    if volume is None:
        raise Exception("volume is None")
    if volume < 2.0:
        raise Exception("pipetting volume below 2.0 is not supported")
    if volume <= 20.0:
        return LABWARE_DICTIONARY["filter_tip_96_20ul"][1]
    if volume <= 300.0:
        return LABWARE_DICTIONARY["tip_96_300ul"][1]
    raise Exception("Max 300 ul tips are supported in this protocol")


class Block:
    """ A full block which includes all the Combinations in this experiment """
    block_size: int
    block_num: int
    starting_index: int
    combinations_list: List[Combination]
    target_plates: List[protocol_api.Labware]

    def __init__(self, block_size:int, block_num: int):
        self.block_size = block_size
        self.block_num = block_num
        self.starting_index = block_num * block_size +1

    def initialize_combinations_list(self, species_list: List[Species], target_plates: List[protocol_api.Labware]):
        """ Initialize the combinations list for one Block with all the possible
         binomial coefficient combinations of the SPECIES_LIST
          into a randomized single-level list """

        self.target_plates = target_plates
        number_of_species = len(species_list)
        combinations_list = []

        # https://stackoverflow.com/a/464882
        # https://docs.python.org/3/library/itertools.html#itertools.combinations
        for i in range(0, number_of_species):
            # print("appending", list(itertools.combinations(SPECIES_LIST, i + 1)))
            combinations_list.append(list(itertools.combinations(species_list, i + 1)))

        combinations_list = flatten_list(combinations_list)
        combinations_list: list

        # test that we have the correct number of combinations (63 combinations with 6 species)
        assert len(combinations_list) == sum(math.comb(number_of_species, x) for x in range(1,
                                                                                            number_of_species + 1)), "combinations_list length should be equal to amount of combinations"
        random.shuffle(combinations_list)  # randomize the order

        collections_list = []

        coords = ["A1", "B1", "C1", "D1", "E1", "F1", "G1",
                  "H1"] * 12  # TODO for testing; do this better, use Opentrons API?
        # TODO or create method that defines the target wells for the current block, and make
        #  easy interface between custom Well and Opentrons Well

        for i, comb in enumerate(combinations_list):
            collections_list.append(Combination(comb, Well(coords[i])))

        # collections_list = list(map(lambda combs: Combination(combs, Well("A1")), combinations_list))

        self.combinations_list = collections_list
        return collections_list

    def transfer_block(self, protocol: protocol_api.ProtocolContext, pipette: protocol_api.InstrumentContext, source_plate: protocol_api.Labware, target_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool):
        for combination in self.combinations_list:
            try:
                combination.pipette_combination(pipette, source_plate, target_plate, total_volume,
                                                change_pipettes)
            except OutOfTipsError:
                protocol.pause("Out of tips for pipette " + str(pipette) + "! Reload all tips back to starting configuration, then resume.")
                pipette.reset_tipracks()
                combination.pipette_combination(pipette, source_plate, target_plate, total_volume,
                                                change_pipettes)





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
 # "kontaminaation ehkäisemiseksi robotin voi määrätä
# koskemaan pipetin kärjellä kaivoin reunaa kun se on pipetoinut nesteen kaivoon"
TOUCH_SIDE_AFTER_DISPENSE = True  # TODO not used yet
# modify here in one place the labware that we are using (1. opentrons API name, 2. volume)
LABWARE_DICTIONARY = {
    'filter_tip_96_20ul': ('opentrons_96_filtertiprack_20ul', 20.0),
    'tip_96_300ul': ('opentrons_96_tiprack_300ul', 300.0),
    'plate_96_200ul': ('biorad_96_wellplate_200ul_pcr', 200.0) # pitää ilmeisesti muuttaa Sarstedtin custom-levyksi
}

BLOCK_SIZE = 64  # wells
CONTROL_WELLS_PER_BLOCK = 1
BLOCKS = [Block(BLOCK_SIZE, 1), Block(BLOCK_SIZE, 2), Block(BLOCK_SIZE, 3)]

####################


def flatten_list(l: list):
    ''' Transform to a single-leveled list
    https://stackoverflow.com/a/952952 '''
    return [item for sublist in l for item in sublist]


assert len(SPECIES_LIST) == NUMBER_OF_SPECIES, "number of species and species list length should be the same"

# https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
metadata = {"apiLevel": "2.13"}  # ylin taso jolla opentrons_simulate toimii


def add_to_required_tip_amounts(comb: Combination, amounts: dict, target_well_volume_ul: float):
    amount, tip_volume = comb.get_required_tips(target_well_volume_ul)
    if tip_volume not in amounts.keys():
        raise Exception("this tip volume is not accepted:", tip_volume)
    amounts[tip_volume] += amount


def run(protocol: protocol_api.ProtocolContext):
    # https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
    tiprack_20ul_1 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 1) # saatetaan tarvita monta
    tiprack_20ul_2 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 2)
    tiprack_20ul_3 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 3)
    tiprack_300ul_1 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 4)  # we need A LOT of tips ...
    tiprack_300ul_2 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 5)
    tiprack_300ul_3 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 6)
    tiprack_300ul_4 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 7)
    tiprack_300ul_5 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 8)

    target_plate1 = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 9)
    target_plate2 = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 10)
    source_plate = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 11)

    required_tip_amounts = {
        20: 0,
        300: 0
    }
    volume_needed_per_species = 0.0
    for block in BLOCKS:
        block_combinations_list: List[Combination]
        block_combinations_list = block.initialize_combinations_list(SPECIES_LIST, [target_plate1, target_plate2])
        for combination in block_combinations_list:
            for s in combination.specie:
                s: Species
                if s.name == SPECIES_1.name:
                    volume_needed_per_species += combination.get_individual_transfer_volume(VOLUME_PER_TARGET_WELL_UL)

            add_to_required_tip_amounts(combination, required_tip_amounts, VOLUME_PER_TARGET_WELL_UL)

    protocol.comment("VOLUME NEEDED PER SPECIES: " + str(volume_needed_per_species))
    protocol.comment("REQUIRED TIP AMOUNTS (volume: amount): " + str(required_tip_amounts))
    protocol.comment("This protocol will pause and request you to load more tips when tips run out")

    # global_combinations_list = initialize_combinations_list(SPECIES_LIST, [target_plate1, target_plate2])



    # https://docs.opentrons.com/v2/new_pipette.html#pipette-models
    left_pipette = protocol.load_instrument("p20_single_gen2", "left", [tiprack_20ul_1, tiprack_20ul_2, tiprack_20ul_3])
    right_pipette = protocol.load_instrument("p300_single_gen2", "right", [tiprack_300ul_1])

    # , tiprack_300ul_2, tiprack_300ul_3, tiprack_300ul_4, tiprack_300ul_5

    # https://docs.opentrons.com/v2/writing.html#commands
    # https://docs.opentrons.com/v2/new_examples.html

    # transfer all the combinations in all the blocks
    for block in BLOCKS:
        block.transfer_block(protocol, right_pipette, source_plate, target_plate1, VOLUME_PER_TARGET_WELL_UL, False)









