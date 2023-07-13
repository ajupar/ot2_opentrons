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
    opentrons_plate: protocol_api.Labware
    opentrons_well: protocol_api.Well
    out_of_liquid: bool  # whether this well is available for transferring

    # https://www.scaler.com/topics/multiple-constructors-python/
    def __init__(self, *args):
        if len(args) == 0:
            self.location = ""
            self.current_volume = 0
            self.maximum_volume = 0
            self.out_of_liquid = True
        elif len(args) == 1:
            self.location = args[0]
            self.current_volume = 0
            self.maximum_volume = 0
            self.out_of_liquid = True
        elif len(args) == 2:
            self.location = args[0]
            self.current_volume = args[1]
            self.maximum_volume = args[1]  # set same as current
            self.out_of_liquid = False
        elif len(args) == 3:
            if args[1] > args[2]:
                raise Exception(self, "Current volume cannot be higher than maximum volume")
            self.location = args[0]
            self.current_volume = args[1]
            self.maximum_volume = args[2]
            self.out_of_liquid = False
        else:
            raise Exception("too many arguments to Well constructor")

    def can_be_transferred_from(self, volume_to_pipette: float):
        if volume_to_pipette + SOURCE_WELL_MARGIN < self.current_volume:
            return True
        if self.out_of_liquid is False:  # comment only on the first occasion to avoid cluttering the output
            protocol_global.comment("Could not transfer from well " + str(self.opentrons_well) + " because current volume (" + str(self.current_volume) + ") is less than the sum of volume to pipette (" + str(volume_to_pipette) + ") and margin volume (" + str(SOURCE_WELL_MARGIN) + "). Proceeding to next available well.")
        self.out_of_liquid = True
        return False

    def get_opentrons_well(self, opentrons_plate: protocol_api.Labware):
        return opentrons_plate[self.location]


class Species:
    """ What we are transferring. Knows from which source
     well to transfer next """
    name: str
    source_wells: list

    def __init__(self, name: str):
        self.name = name

    def get_current_source_well(self, volume_to_transfer: float):
        """ get the first source well with enough fluid, throws exception if no well with enough fluid is found """
        for well in self.source_wells:
            well: Well
            if well.can_be_transferred_from(volume_to_transfer):
                return well
        raise Exception("Could not find source well with enough fluid for Species", self.name, ". Should more source wells be added? Number of source wells", len(self.source_wells), "amount of fluid in last well:", self.source_wells[-1])

    def get_source_wells_total_volume(self) -> float:
        total_volume: float
        total_volume = 0
        for well in self.source_wells:
            total_volume += well.current_volume

        return total_volume



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

    def pipette_combination(self, protocol: protocol_api.ProtocolContext, pipette: protocol_api.InstrumentContext, source_plate: protocol_api.Labware, target_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool, touch_tip: bool):
        """ Calls the Opentrons transfer method to pipette the Species contained in this Combination """
        global DEBUG_TRANSFER_COUNTER
        assert len(self.specie) > 0, "specie of combination not initialized"
        spec: Species
        for spec in self.specie:
            assert spec.name is not None and spec.source_wells is not None

        for spec in self.specie:
            try:
                volume_to_transfer = total_volume / len(self.specie)
                source_well: Well
                source_well = spec.get_current_source_well(volume_to_transfer)
                initial_volume = source_well.current_volume
                protocol.comment("Transferring " + str(volume_to_transfer) + " μl of species " + spec.name + " from well " + str(source_well.get_opentrons_well(source_plate)))
                pipette.transfer(volume_to_transfer, source_well.get_opentrons_well(source_plate), self.target_well.opentrons_well, trash=change_pipettes, touch_tip=touch_tip)
                DEBUG_TRANSFER_COUNTER += 1  # this should run only if no exception is thrown
                source_well.current_volume -= volume_to_transfer
                assert source_well.current_volume == initial_volume - volume_to_transfer, "Source well volume should decrease correctly"
            except OutOfTipsError:
                protocol.pause("Out of tips for pipette " + str(pipette) + " while transferring species " + spec.name + " from well " + str(source_well.get_opentrons_well(source_plate)) + " to well " + str(self.target_well.opentrons_well) +"! Reload all tips back to starting configuration, then resume.")
                pipette.reset_tipracks()

                protocol.comment("Transferring " + str(volume_to_transfer) + " μl of species " + spec.name + " from well " + str(source_well.get_opentrons_well(source_plate)))
                pipette.transfer(volume_to_transfer, source_well.get_opentrons_well(source_plate),
                                 self.target_well.opentrons_well, trash=change_pipettes, touch_tip=touch_tip)
                DEBUG_TRANSFER_COUNTER += 1
                source_well.current_volume -= volume_to_transfer
                assert source_well.current_volume == initial_volume - volume_to_transfer, "Source well volume should decrease correctly"



def define_tip_volume(volume: float) -> float:
    """ Get the right pipette volume for this actual transfer volume.
    TODO for further use in other protocols, this method should be made more generic
     """
    if volume is None:
        raise Exception("volume is None")
    if volume < 2.0:
        raise Exception("pipetting volume below 2.0 is not supported")
    if volume <= 20.0:
        return LABWARE_DICTIONARY["filter_tip_96_20ul"][1]
    if volume <= 300.0:
        return LABWARE_DICTIONARY["tip_96_300ul"][1]
    raise Exception("Max 300 ul tips are supported in this protocol")


def define_pipette_to_use(pipette_volume: float, pipettes: List[protocol_api.InstrumentContext]):
    """ Get the pipette to use from a two-member list of pipettes with 20.0 ul pipette at index 0
    and 300.0 ul pipette at index 1. The pipette-volume parameter must be either 20.0 or 300.0.
    TODO for further use in other protocols, this method should be made more generic """
    if len(pipettes) != 2:
        raise Exception("A two-member list of Opentrons Pipette objects must be provided for defining the pipette to use")
    if pipette_volume == 20.0:
        return pipettes[0]
    if pipette_volume == 300.0:
        return pipettes[1]
    raise Exception("This protocol can define the pipette only for 20.0 and 300.0 pipette volumes")


class Block:
    """ A full block which includes all the Combinations in this experiment """
    block_size: int
    block_num: int
    starting_index: int
    control_wells_amount: int
    combinations_list: List[Combination]
    target_plates: List[protocol_api.Labware]

    def __init__(self, block_size:int, block_num: int, control_wells_amount: int):
        self.block_size = block_size
        self.block_num = block_num
        self.control_wells_amount = control_wells_amount
        self.starting_index = (block_num-1) * block_size

    def define_current_target_plate_and_well(self, index_inside_block: int, target_plates: List[protocol_api.Labware]) -> Tuple[protocol_api.Labware, protocol_api.Well]:
        """ Return the Opentrons plate and Opentrons Well for this transfer """
        if self.starting_index is None:
            raise Exception("starting_index should be defined")
        current_index = self.starting_index + index_inside_block
        for plate in target_plates:
            if current_index < len(plate.wells()):
                return plate, plate.well(current_index)
            else:
                current_index -= len(plate.wells())
                continue
        raise Exception("Could not define target plate, are enough target plates available?")

    def initialize_combinations_list(self, species_list: List[Species], target_plates: List[protocol_api.Labware]):
        """ Initialize the combinations list for one Block with all the possible
         binomial coefficient combinations of the SPECIES_LIST
          into a randomized single-level list """

        self.target_plates = target_plates
        number_of_species = len(species_list)
        species_combinations_list = []

        # https://stackoverflow.com/a/464882
        # https://docs.python.org/3/library/itertools.html#itertools.combinations
        for i in range(0, number_of_species):
            # print("appending", list(itertools.combinations(SPECIES_LIST, i + 1)))
            species_combinations_list.append(list(itertools.combinations(species_list, i + 1)))

        species_combinations_list = flatten_list(species_combinations_list)
        species_combinations_list: list

        # test that we have the correct number of combinations (63 combinations with 6 species)
        assert len(species_combinations_list) == sum(math.comb(number_of_species, x) for x in range(1,
                                                                                            number_of_species + 1)), "combinations_list length should be equal to amount of combinations"

        assert len(species_combinations_list) + self.control_wells_amount == self.block_size, "Block size should equal amount of combinations + control wells"

        # add controls, which are also coded as Species for simplicity
        for i in range(0, self.control_wells_amount):
            species_combinations_list.append([CONTROL])  # create single-member lists because control includes only one "species", ie. type of fluid

        random.shuffle(species_combinations_list)  # randomize the order

        assert len(species_combinations_list) == self.block_size, "After adding controls, block size should equal amount of combinations"

        combination_objects_list = []

        # find Opentrons target plate and well for each combination,
        # then construct them as Combination objects for easier further handling
        for i, comb in enumerate(species_combinations_list):
            current_target_plate, current_target_well = self.define_current_target_plate_and_well(i, target_plates)
            target_well = Well()
            target_well.opentrons_plate = current_target_plate
            target_well.opentrons_well = current_target_well
            combination_objects_list.append(Combination(comb, target_well))

        self.combinations_list = combination_objects_list

        return combination_objects_list

    def transfer_block(self, protocol: protocol_api.ProtocolContext, pipettes: List[protocol_api.InstrumentContext], source_plate: protocol_api.Labware, target_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool, touch_tip: bool):
        """ Transfer the combinations and controls """
        protocol.comment("Starting to transfer block " + str(self.block_num) + " with " + str(len(self.combinations_list)-self.control_wells_amount) + " combination(s) and " + str(self.control_wells_amount) + " control well(s)")

        for index, combination in enumerate(self.combinations_list):
            protocol.comment("Starting to transfer combination " + str(index+1) + " in block " + str(self.block_num) + ", with species " + str([x.name for x in combination.specie]) + " into well " + str(combination.target_well.opentrons_well))
            pipette: protocol_api.InstrumentContext
            individual_transfer_volume = combination.get_individual_transfer_volume(total_volume)
            defined_tip_volume = define_tip_volume(individual_transfer_volume)
            pipette = define_pipette_to_use(defined_tip_volume, pipettes)
            protocol.comment("Selected pipette " + str(pipette) + " for this combination with transfer volume of " + str(individual_transfer_volume) + " μl per species")
            # to reduce risk of errors, program volume ranges and opentrons pipette volume ranges should be equal
            assert defined_tip_volume == pipette.max_volume, "Defined tip volume in the protocol and the Opentrons pipette max volume should be equal, assuming that the volume ranges defined in the program are identical to the Opentrons pipette volume ranges"
            combination.pipette_combination(protocol, pipette, source_plate, target_plate, total_volume, change_pipettes, touch_tip)


def generate_source_wells(column: int, number_of_rows: int, initial_volume: float, source_plate: protocol_api.Labware) -> List[Well]:
    """ Generate source wells, a given number of rows on a given column. Also bind the wells
     to the given Opentrons plate and corresponding well """
    wells: List[Well]
    wells = []
    for i in range(1, number_of_rows+1):
        coordinate = str(dict_rownumber_letter[i] + str(column))
        well = Well(coordinate, initial_volume)
        well.opentrons_plate = source_plate
        well.opentrons_well = source_plate[coordinate]
        wells.append(well)

    return wells


dict_rownumber_letter = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H"
}

##########################
# This part can be modified by the user:
# script starting values
SOURCE_WELLS_INITIAL_VOLUME_UL = 1000.0  # modify this based on initial volume; affects how source well is changed
SOURCE_WELL_MARGIN = 50.0  # how many ul is left to source wells before changing to another
NUMBER_OF_SOURCE_WELLS_PER_SPECIES = 3
NUMBER_OF_CONTROL_WELLS = 1
# 1: name, 2: list of source wells (incl well location and fluid volume)
SPECIES_1 = Species("laji1")  # source wells need to be generated inside run() method because they need access to the Opentrons plate
SPECIES_2 = Species("laji2")
SPECIES_3 = Species("laji3")
SPECIES_4 = Species("laji4")
SPECIES_5 = Species("laji5")
SPECIES_6 = Species("laji6")
# program logic handles control as a Species object
CONTROL = Species("CONTROL")
SPECIES_LIST = [SPECIES_1, SPECIES_2, SPECIES_3, SPECIES_4, SPECIES_5, SPECIES_6]
NUMBER_OF_SPECIES = len(SPECIES_LIST)  # trivial but maybe useful to see
VOLUME_PER_TARGET_WELL_UL = 60.0
CHANGE_TIPS = True  # change tips after each transfer; almost always should be true
 # "kontaminaation ehkäisemiseksi robotin voi määrätä
# koskemaan pipetin kärjellä kaivoin reunaa kun se on pipetoinut nesteen kaivoon"
# https://docs.opentrons.com/v2/new_complex_commands.html#touch-tip
TOUCH_TIP = True  #  extra precaution to avoid contamination
# modify here in one place the labware that we are using (1. opentrons API name, 2. volume)
LABWARE_DICTIONARY = {
    'filter_tip_96_20ul': ('opentrons_96_filtertiprack_20ul', 20.0),
    'tip_96_300ul': ('opentrons_96_tiprack_300ul', 300.0),
    'plate_96_200ul': ('biorad_96_wellplate_200ul_pcr', 200.0) # pitää ilmeisesti muuttaa Sarstedtin custom-levyksi
}

BLOCK_SIZE = 64  # wells
CONTROL_WELLS_PER_BLOCK = 1
BLOCKS = [Block(BLOCK_SIZE, 1, 1), Block(BLOCK_SIZE, 2, 1), Block(BLOCK_SIZE, 3, 1)]


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


def define_source_wells(source_plate: protocol_api.Labware):
    """ Populate source wells and bind them to the Opentrons source plate """
    SPECIES_1.source_wells = generate_source_wells(1, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)
    SPECIES_2.source_wells = generate_source_wells(2, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)
    SPECIES_3.source_wells = generate_source_wells(3, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)
    SPECIES_4.source_wells = generate_source_wells(4, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)
    SPECIES_5.source_wells = generate_source_wells(5, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)
    SPECIES_6.source_wells = generate_source_wells(6, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)
    CONTROL.source_wells = generate_source_wells(7, NUMBER_OF_CONTROL_WELLS, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)



def run(protocol: protocol_api.ProtocolContext):
    # iterates the number of actually executed transfers for quality assurance purposes,
    # will be asserted against the pre-calculated theoretical amount of transfers
    global DEBUG_TRANSFER_COUNTER
    DEBUG_TRANSFER_COUNTER = 0

    global protocol_global
    protocol_global = protocol  # needed for protocol.comment access inside modules that otherwise don't use protocol

    # https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
    # we need a lot of tips, but the program also prompts tip change when needed
    tiprack_20ul_1 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 1)
    tiprack_20ul_2 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 2)
    tiprack_20ul_3 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 3)
    tiprack_300ul_1 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 4)
    tiprack_300ul_2 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 5)
    tiprack_300ul_3 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 6)
    tiprack_300ul_4 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 7)
    # tiprack_300ul_5 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 8)

    # ensure there are enough target plates for the amount of combinations
    target_plate1 = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 8)
    target_plate2 = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 9)
    target_plate3 = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 10)
    # assuming that one is enough
    source_plate = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 11)

    define_source_wells(source_plate)

    # calculate theoretical tip usage, used for quality testing
    required_tip_amounts = {
        20: 0,
        300: 0
    }
    volume_needed_per_species = 0.0
    for block in BLOCKS:
        block_combinations_list: List[Combination]
        block_combinations_list = block.initialize_combinations_list(SPECIES_LIST, [target_plate1, target_plate2, target_plate3])
        for combination in block_combinations_list:
            for s in combination.specie:
                s: Species
                if s.name == SPECIES_1.name:
                    volume_needed_per_species += combination.get_individual_transfer_volume(VOLUME_PER_TARGET_WELL_UL)

            add_to_required_tip_amounts(combination, required_tip_amounts, VOLUME_PER_TARGET_WELL_UL)

    protocol.comment("VOLUME NEEDED PER SPECIES: " + str(volume_needed_per_species))
    protocol.comment("REQUIRED TIP AMOUNTS (volume: amount): " + str(required_tip_amounts) + ", in total: " + str(sum(required_tip_amounts.values())))
    protocol.comment("This protocol will pause and request you to load more tips when tips run out")

    # https://docs.opentrons.com/v2/new_pipette.html#pipette-models
    left_pipette_20ul = protocol.load_instrument("p20_single_gen2", "left", [tiprack_20ul_1]) # in simulation, not using more than one tip rack per type to demonstrate the tip change prompt mechanism
    right_pipette_300ul = protocol.load_instrument("p300_single_gen2", "right", [tiprack_300ul_1])

    # https://docs.opentrons.com/v2/writing.html#commands
    # https://docs.opentrons.com/v2/new_examples.html

    # transfer all the combinations in all the blocks
    for block in BLOCKS:
        # for simplicity, pipettes are provided as a 2-member list with the smaller volume always at index 0 and higher volume at index 1
        # TODO for wider use in other protocols, this should be done in a smarter and more generic way
        block.transfer_block(protocol, [left_pipette_20ul, right_pipette_300ul], source_plate, target_plate1, VOLUME_PER_TARGET_WELL_UL, CHANGE_TIPS, TOUCH_TIP)


    # test for protocol's logical integrity
    assert sum(required_tip_amounts.values()) == DEBUG_TRANSFER_COUNTER, "pre-calculated total tip usage should equal the amount of executed transfers"

    protocol.comment("Amount of executed transfers at the end is " + str(DEBUG_TRANSFER_COUNTER) + ", total pre-calculated tip consumption was " + str(sum(required_tip_amounts.values())) + ", values are equal: " + str(DEBUG_TRANSFER_COUNTER == sum(required_tip_amounts.values())))







