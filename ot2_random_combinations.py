# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from io import TextIOWrapper
from typing import List, Tuple
from enum import Enum
from opentrons import protocol_api
import itertools
import random
import csv
import platform
from datetime import datetime
from math import factorial as fact
from opentrons.protocol_api.labware import OutOfTipsError


def binomial(a,b):
    """" https://www.pythonpool.com/python-binomial-coefficient/ """
    return fact(a) // fact(b) // fact(a-b)


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

    def pipette_combination(self, protocol: protocol_api.ProtocolContext, pipette: protocol_api.InstrumentContext, source_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool, touch_tip: bool, block_num: int, comb_num: int):
        """ Calls the Opentrons transfer method to pipette the Species contained in this Combination """
        global DEBUG_TRANSFER_COUNTER
        assert len(self.specie) > 0, "specie of combination not initialized"
        spec: Species
        for spec in self.specie:
            assert spec.name is not None and spec.source_wells is not None

        volume_to_transfer = total_volume / len(self.specie)

        for spec in self.specie:
            try:

                source_well: Well
                source_well = spec.get_current_source_well(volume_to_transfer)
                assert source_well.location == source_well.get_opentrons_well(source_plate).well_name, "Source custom well and Opentrons well location should match"
                # target well location is defined only for Opentrons well, see Block.define_current_target_plate_and_well() usage
                initial_volume = source_well.current_volume
                protocol.comment("Transferring " + str(volume_to_transfer) + " μl of species " + spec.name + " from well " + str(source_well.get_opentrons_well(source_plate)) + " to " + str(self.target_well.opentrons_well))
                pipette.transfer(volume_to_transfer, source_well.get_opentrons_well(source_plate), self.target_well.opentrons_well, trash=change_pipettes, touch_tip=touch_tip)
                DEBUG_TRANSFER_COUNTER += 1  # this should run only if no exception is thrown
                source_well.current_volume -= volume_to_transfer
                assert source_well.current_volume == initial_volume - volume_to_transfer, "Source well volume should decrease correctly"

            except OutOfTipsError:
                protocol.pause("Out of tips for pipette " + str(pipette) + " while transferring species " + spec.name + " from well " + str(source_well.get_opentrons_well(source_plate)) + " to well " + str(self.target_well.opentrons_well) +"! Reload the tips for this pipette back to starting configuration, then resume.")
                pipette.reset_tipracks()

                protocol.comment("Transferring " + str(volume_to_transfer) + " μl of species " + spec.name + " from well " + str(source_well.get_opentrons_well(source_plate)) + " to " + str(self.target_well.opentrons_well))
                pipette.transfer(volume_to_transfer, source_well.get_opentrons_well(source_plate),
                                 self.target_well.opentrons_well, trash=change_pipettes, touch_tip=touch_tip)
                DEBUG_TRANSFER_COUNTER += 1
                source_well.current_volume -= volume_to_transfer
                assert source_well.current_volume == initial_volume - volume_to_transfer, "Source well volume should decrease correctly"

        FileHandler.write_row(FileType.TRANSFERS, [block_num, comb_num, str([x.name for x in self.specie]), len(self.specie), volume_to_transfer, total_volume, str(self.target_well.opentrons_plate), self.target_well.opentrons_well.well_name])


def define_tip_volume(volume: float) -> float:
    """ Get the right pipette volume for this actual transfer volume.
    TODO for further use in other protocols, this method should be made more generic
     """
    if volume is None:
        raise Exception("volume is None")
    if volume < 2.0:
        raise Exception("pipetting volume below 2.0 is not supported")
    if volume < 20.0:  # NOTE 20 ul is pipetted with larger pipette to allow using only one pipette for the whole experiment with volume range 20-120 ul
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
    local_species_list: List[Species]
    controls: List[Species]
    block_size: int
    block_num: int
    starting_index: int
    combinations_list: List[Combination]
    target_plates: List[protocol_api.Labware]

    def __init__(self, species_list: List[Species], controls: List[Species], block_size: int, block_num: int):
        self.local_species_list = species_list
        self.controls = controls
        self.block_size = block_size
        self.block_num = block_num
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

    def initialize_combinations_list(self, block_num: int, target_plates: List[protocol_api.Labware]):
        """ Initialize the combinations list for one Block with all the possible
         binomial coefficient combinations of the SPECIES_LIST
          into a randomized single-level list. block_num here is needed here as a coefficient
           for the random seed so that the order within each block is different """

        self.target_plates = target_plates
        number_of_species = len(self.local_species_list)
        species_combinations_list = []

        # https://stackoverflow.com/a/464882
        # https://docs.python.org/3/library/itertools.html#itertools.combinations
        for i in range(0, number_of_species):
            # print("appending", list(itertools.combinations(SPECIES_LIST, i + 1)))
            species_combinations_list.append(list(itertools.combinations(self.local_species_list, i + 1)))

        species_combinations_list = flatten_list(species_combinations_list)
        species_combinations_list: list

        # test that we have the correct number of combinations (63 combinations with 6 species)
        assert len(species_combinations_list) == sum(binomial(number_of_species, x) for x in range(1,
                                                                                            number_of_species + 1)), "combinations_list length should be equal to amount of combinations"

        assert len(species_combinations_list) + len(self.controls) == self.block_size, "Block size should equal amount of combinations + control wells"

        # add controls, which are also coded as Species for simplicity
        for control in self.controls:
            species_combinations_list.append([control])  # create single-member lists because control includes only one "species", ie. type of fluid

        # random seed is needed to ensure that the order is the same in Opentrons App and the robot
        # using block_num as coefficient for random seed to ensure that order within each block is different
        random.Random(RANDOM_SEED * block_num).shuffle(species_combinations_list)  # randomize the order

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

        self.assert_species_appearances()

        return combination_objects_list

    def assert_species_appearances(self):
        """ Quality assurance method. Each species should appear in the combinations of a block the number of times defined by binomial coefficient calculations """
        for spec in self.local_species_list:
            appearances = self.count_species_appearances(spec)
            # protocol_global.comment("Species " + spec.name + " appears " + str(appearances) + " times in Block " + str(self.block_num) + ", expected appearances: " + str(self.get_expected_appearances((len(self.local_species_list)))))
            assert appearances == self.get_expected_appearances(len(self.local_species_list)), "each species should appear in the combinations of a block the number of times defined by binomial coefficient calculations"

    def count_species_appearances(self, spec1: Species) -> int:
        """ Quality assurance helper method """
        count = 0

        for combination in self.combinations_list:
            for spec2 in combination.specie:
                spec2: Species
                if spec2.name == spec1.name:
                    count += 1

        return count

    def get_expected_appearances(self, n: int):
        """ For quality assurance, get the amount of times a species should appear in the combinations """
        expected = 0

        for k in range(1, (n+1)):
            if k < n:
                expected += (binomial(n, k) - binomial(n-1, k))  # subtract combinations where the species shouldn't appear
            elif k == n:
                expected += binomial(n, k)  # this is always just 1

        return expected

    def transfer_block(self, protocol: protocol_api.ProtocolContext, pipettes: List[protocol_api.InstrumentContext], source_plate: protocol_api.Labware, target_plate: protocol_api.Labware, total_volume: float, change_pipettes: bool, touch_tip: bool):
        """ Transfer the combinations and controls """
        protocol.comment("Starting to transfer block " + str(self.block_num) + " with " + str(len(self.combinations_list)-len(self.controls)) + " combination(s) and " + str(len(self.controls)) + " control well(s), with species " + str([x.name for x in self.local_species_list]) + " and controls " + str([x.name for x in self.controls]))

        for index, combination in enumerate(self.combinations_list):
            protocol.comment("Starting to transfer combination " + str(index+1) + " in block " + str(self.block_num) + ", with species " + str([x.name for x in combination.specie]) + " into well " + str(combination.target_well.opentrons_well))
            pipette: protocol_api.InstrumentContext
            individual_transfer_volume = combination.get_individual_transfer_volume(total_volume)
            defined_tip_volume = define_tip_volume(individual_transfer_volume)
            pipette = define_pipette_to_use(defined_tip_volume, pipettes)
            protocol.comment("Selected pipette " + str(pipette) + " for this combination with transfer volume of " + str(individual_transfer_volume) + " μl per species")
            # to reduce risk of errors, program volume ranges and opentrons pipette volume ranges should be equal
            assert defined_tip_volume == pipette.max_volume, "Defined tip volume in the protocol and the Opentrons pipette max volume should be equal, assuming that the volume ranges defined in the program are identical to the Opentrons pipette volume ranges"
            combination.pipette_combination(protocol, pipette, source_plate, total_volume, change_pipettes, touch_tip, block_num=self.block_num, comb_num=(index+1))


def generate_source_wells(spec: Species, column_index: int, row_index: int, number_of_rows: int, initial_volume: float, source_plate: protocol_api.Labware) -> List[Well]:
    """ Generate source wells, a given number of rows on a given column starting from the given row. Also bind the wells
     to the given Opentrons plate and corresponding well """
    wells: List[Well]
    wells = []
    for i in range(row_index, row_index + number_of_rows):
        coordinate = str(dict_rownumber_letter[i] + str(column_index))
        well = Well(coordinate, initial_volume)
        well.opentrons_plate = source_plate
        well.opentrons_well = source_plate[coordinate]
        wells.append(well)

        FileHandler.write_row(FileType.SOURCES, [spec.name, str(well.opentrons_plate), well.opentrons_well.well_name, SOURCE_WELLS_INITIAL_VOLUME_UL])


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
LINUX_COMPUTER_NAME = "V155-15API"  # get this with platform.node(). used to separate analyses/runs on computer vs robot
MAC_COMPUTER_NAME = "TY2302039"  # Robot room's Mac computer
# https://docs.opentrons.com/v2/robot_position.html#gantry-speed
# https://docs.opentrons.com/v2/new_protocol_api.html?highlight=speed#opentrons.protocol_api.InstrumentContext.default_speed
# "These default speeds were chosen because they’re the maximum speeds that Opentrons knows will work with the gantry.
# Your robot may be able to move faster, but you shouldn’t increase this value unless instructed by Opentrons Support."
GANTRY_SPEED = 600.0  # default is 400.0.
FLOW_RATE_20ul = 9.8  # default 7.56  -> 7.56 * 1.3
FLOW_RATE_300ul = 120.7  # default 92.86  -> 92.86 * 1.3
RANDOM_SEED = 18  # use static seed to get same order in Opentrons App and the robot, because both run the protocol independendtly in forming the protocol steps
SOURCE_WELLS_INITIAL_VOLUME_UL = 1000.0  # modify this based on initial volume; affects how source well is changed
SOURCE_WELL_MARGIN = 50.0  # how many ul is left to source wells before changing to another
NUMBER_OF_SOURCE_WELLS_PER_SPECIES = 2
NUMBER_OF_CONTROL_WELLS = 1
# 1: name, 2: list of source wells (incl well location and fluid volume)
SPECIES_1 = Species("laji1")  # source wells need to be generated inside run() method because they need access to the Opentrons plate
SPECIES_2 = Species("laji2")
SPECIES_3 = Species("laji3")
SPECIES_4 = Species("laji4")
SPECIES_5 = Species("laji5")
SPECIES_6 = Species("laji6")
SPECIES_7 = Species("laji7")
SPECIES_8 = Species("laji8")
SPECIES_9 = Species("laji9")
SPECIES_10 = Species("laji10")
SPECIES_11 = Species("laji11")
SPECIES_12 = Species("laji12")
SPECIES_13 = Species("laji13")
SPECIES_14 = Species("laji14")
SPECIES_15 = Species("laji15")
SPECIES_16 = Species("laji16")
SPECIES_17 = Species("laji17")
SPECIES_18 = Species("laji18")
# program logic handles control as a Species object
CONTROLS_BLOCK_1 = [Species("CONTROL_BLOCK_1")]
CONTROLS_BLOCK_2 = [Species("CONTROL_BLOCK_2")]
CONTROLS_BLOCK_3 = [Species("CONTROL_BLOCK_3")]
ALL_CONTROLS = CONTROLS_BLOCK_1 + CONTROLS_BLOCK_2 + CONTROLS_BLOCK_3
BLOCK_1_SPECIES = [SPECIES_1, SPECIES_2, SPECIES_3, SPECIES_4, SPECIES_5, SPECIES_6]
BLOCK_2_SPECIES = [SPECIES_7, SPECIES_8, SPECIES_9, SPECIES_10, SPECIES_11, SPECIES_12]
BLOCK_3_SPECIES = [SPECIES_13, SPECIES_14, SPECIES_15, SPECIES_16, SPECIES_17, SPECIES_18]
ALL_SPECIES = BLOCK_1_SPECIES + BLOCK_2_SPECIES + BLOCK_3_SPECIES
NUMBER_OF_SPECIES = len(BLOCK_1_SPECIES)  # trivial but maybe useful to see
VOLUME_PER_TARGET_WELL_UL = 120.0
CHANGE_TIPS = False  # change tips after each transfer; almost always should be true
 # "kontaminaation ehkäisemiseksi robotin voi määrätä
# koskemaan pipetin kärjellä kaivoin reunaa kun se on pipetoinut nesteen kaivoon"
# https://docs.opentrons.com/v2/new_complex_commands.html#touch-tip
TOUCH_TIP = True  #  extra precaution to avoid contamination
# modify here in one place the labware that we are using (1. opentrons API name, 2. volume)
LABWARE_DICTIONARY = {
    'filter_tip_96_20ul': ('opentrons_96_filtertiprack_20ul', 20.0),
    'tip_96_300ul': ('opentrons_96_tiprack_300ul', 300.0),
    'plate_96_200ul': ('biorad_96_wellplate_200ul_pcr', 200.0),  # pitää ilmeisesti muuttaa Sarstedtin custom-levyksi
    'tipone_96_200ul': ("tipone_96_tiprack_200ul", 200.0),
    'juhani_deepwell_plate': ('sarstedt_96_wellplate_2200ul', 2000.0)  # custom määrittely
}

BLOCK_SIZE = 16  # lyhyt testiajo
# BLOCK_SIZE = 64  # kokonainen ajo
CONTROL_WELLS_PER_BLOCK = 1

# kokonainen testiajo
# BLOCKS = [Block(BLOCK_1_SPECIES, CONTROLS_BLOCK_1, BLOCK_SIZE, 1), Block(BLOCK_2_SPECIES, CONTROLS_BLOCK_2, BLOCK_SIZE, 2), Block(BLOCK_3_SPECIES, CONTROLS_BLOCK_3, BLOCK_SIZE, 3)]
# BLOCKS = [Block(BLOCK_1_SPECIES, CONTROLS_BLOCK_1, BLOCK_SIZE, 1)]
# lyhyt testiajo:
BLOCKS = [Block(BLOCK_1_SPECIES[0:4], CONTROLS_BLOCK_1, BLOCK_SIZE, 1)]


####################


def flatten_list(l: list):
    ''' Transform to a single-leveled list
    https://stackoverflow.com/a/952952 '''
    return [item for sublist in l for item in sublist]


assert len(BLOCK_1_SPECIES) == NUMBER_OF_SPECIES, "number of species and species list length should be the same"

# https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
metadata = {"apiLevel": "2.13"}  # ylin taso jolla opentrons_simulate toimii


def add_to_required_tip_amounts(comb: Combination, amounts: dict, target_well_volume_ul: float):
    amount, tip_volume = comb.get_required_tips(target_well_volume_ul)
    if tip_volume not in amounts.keys():
        raise Exception("this tip volume is not accepted:", tip_volume)
    amounts[tip_volume] += amount


def define_source_wells(source_plate: protocol_api.Labware):
    """ Populate source wells and bind them to the Opentrons source plate """

    # all_sources = ALL_SPECIES + ALL_CONTROLS
    all_sources = BLOCK_1_SPECIES + CONTROLS_BLOCK_1 + BLOCK_2_SPECIES + CONTROLS_BLOCK_2 + BLOCK_3_SPECIES + CONTROLS_BLOCK_3

    for i in range(0, len(all_sources)):
        # two per one column, at row indexes 1 and 5
        # for 18 species, this means columns 1-9 are used
        # with 3 additional controls, this is 21 sources in total which uses 10.5 columns
        define_sources_with_defaults(all_sources[i], ((i // 2) + 1), (1 if (i % 2 == 0) else 5), source_plate)


def define_sources_with_defaults(spec: Species, colIndex: int, rowIndex: int, source_plate: protocol_api.Labware):
    spec.source_wells = generate_source_wells(spec, colIndex, rowIndex, NUMBER_OF_SOURCE_WELLS_PER_SPECIES, SOURCE_WELLS_INITIAL_VOLUME_UL, source_plate)


class FileType(Enum):
    """ https://realpython.com/python-enum/ """
    SOURCES = "sources"
    TRANSFERS = "transfers"


class FileHandler:
    """ https://www.digitalocean.com/community/tutorials/python-static-method  <br/> <br/>
    Keep the file writers open from the beginning to save any written data even if
     the protocol is aborted before it ends
    """
    file_path: str
    sources_filename: str
    transfers_filename: str
    sources_file: TextIOWrapper
    transfers_file: TextIOWrapper
    transfers_writer: None
    sources_writer: None
    csv_transfers_header = ["block", "combination", "species", "n_species", "individ_transf_vol", "total_transf_vol",
                            "destination_plate", "destination_well"]
    csv_sources_header = ["species", "source_plate", "source_well", "starting_volume_ul"]

    @staticmethod
    def init_writers():
        """  https://www.pythontutorial.net/python-basics/python-write-csv-file/ """
        FileHandler.file_path = FileHandler.define_file_path()
        FileHandler.sources_filename = FileHandler.generate_filename(FileType.SOURCES.value)
        FileHandler.transfers_filename = FileHandler.generate_filename(FileType.TRANSFERS.value)

        FileHandler.sources_file = open(FileHandler.file_path + FileHandler.sources_filename, "w")
        FileHandler.transfers_file = open(FileHandler.file_path + FileHandler.transfers_filename, "w")

        FileHandler.sources_writer = csv.writer(FileHandler.sources_file)
        FileHandler.transfers_writer = csv.writer(FileHandler.transfers_file)

    @staticmethod
    def write_header_rows():
        FileHandler.write_row(FileType.SOURCES, FileHandler.csv_sources_header)
        FileHandler.write_row(FileType.TRANSFERS, FileHandler.csv_transfers_header)

    @staticmethod
    def define_file_path() -> str:
        """ Different paths are needed when running on computer vs. robot """
        if platform.node() == LINUX_COMPUTER_NAME:  # https://stackoverflow.com/a/4271873
            return "/home/atte/Ohjelmointi/pycharm-workspace/OT2_random_combinations/output/"  # running in computer
        elif platform.node() == MAC_COMPUTER_NAME:
            return "Users/biologia/Desktop/Juhani_koe1/"
        else:
            return "/data/user_storage/"  # running inside robot

    @staticmethod
    def write_row(filetype: Enum, row: List[str]):
        """ filetype should be 'transfers' or 'sources' """
        if filetype == FileType.TRANSFERS:
            FileHandler.transfers_writer.writerow(row)
        elif filetype == FileType.SOURCES:
            FileHandler.sources_writer.writerow(row)
        else:
            raise Exception("Invalid filetype for FileHandler.write_row(), must be 'transfers' or 'sources'. Input was " + str(filetype))

    @staticmethod
    def generate_filename(identifier: str) -> str:
        return identifier + "_" + datetime.now().strftime("%d%m%Y_%H-%M-%S") + ".csv"

    @staticmethod
    def close_files():
        FileHandler.sources_file.close()
        FileHandler.transfers_file.close()


def run(protocol: protocol_api.ProtocolContext):
    FileHandler.init_writers()
    FileHandler.write_header_rows()

    # iterates the number of actually executed transfers for quality assurance purposes,
    # will be asserted against the pre-calculated theoretical amount of transfers
    global DEBUG_TRANSFER_COUNTER
    DEBUG_TRANSFER_COUNTER = 0

    global protocol_global
    protocol_global = protocol  # needed for protocol.comment access inside modules that otherwise don't use protocol

    # https://docs.opentrons.com/v2/writing.html#the-run-function-and-the-protocol-context
    # we need a lot of tips, but the program also prompts tip change when needed
    # tiprack_20ul_1 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 10)
    # tiprack_20ul_2 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 2)
    # tiprack_20ul_3 = protocol.load_labware(LABWARE_DICTIONARY["filter_tip_96_20ul"][0], 3)
    tiprack_300ul_1 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 7)
    # tiprack_300ul_1 = protocol.load_labware(LABWARE_DICTIONARY["tipone_96_200ul"], 7)

    # tiprack_300ul_2 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 5)
    # tiprack_300ul_3 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 6)
    # tiprack_300ul_4 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 7)
    # tiprack_300ul_5 = protocol.load_labware(LABWARE_DICTIONARY["tip_96_300ul"][0], 8)

    # ensure there are enough target plates for the amount of combinations
    target_plate1 = protocol.load_labware(LABWARE_DICTIONARY["juhani_deepwell_plate"][0], 2)
    target_plate2 = protocol.load_labware(LABWARE_DICTIONARY["juhani_deepwell_plate"][0], 3)
    # target_plate3 = protocol.load_labware(LABWARE_DICTIONARY["plate_96_200ul"][0], 10)
    # assuming that one is enough
    source_plate = protocol.load_labware(LABWARE_DICTIONARY["juhani_deepwell_plate"][0], 4)

    define_source_wells(source_plate)

    # calculate theoretical tip usage, used for quality testing
    required_tip_amounts = {
        20: 0,
        300: 0
    }
    volume_needed_per_species = 0.0
    for block in BLOCKS:
        block_combinations_list: List[Combination]
        block_combinations_list = block.initialize_combinations_list(block.block_num, [target_plate1, target_plate2])  # removed target_plate3
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
    # left_pipette_20ul = protocol.load_instrument("p20_single_gen2", "left", [tiprack_20ul_1]) # in simulation, not using more than one tip rack per type to demonstrate the tip change prompt mechanism
    right_pipette_300ul = protocol.load_instrument("p300_single_gen2", "right", [tiprack_300ul_1])

    # default is 400
    # left_pipette_20ul.default_speed = GANTRY_SPEED  # https://docs.opentrons.com/v2/new_protocol_api.html?highlight=speed#opentrons.protocol_api.InstrumentContext.default_speed
    right_pipette_300ul.default_speed = GANTRY_SPEED

    # https://docs.opentrons.com/v2/new_pipette.html#ot-2-pipette-flow-rates
    # default 7.56
    # left_pipette_20ul.flow_rate.aspirate = FLOW_RATE_20ul  # default * 1.3
    # left_pipette_20ul.flow_rate.dispense = FLOW_RATE_20ul
    # left_pipette_20ul.flow_rate.blow_out = FLOW_RATE_20ul
    # default 92.86
    right_pipette_300ul.flow_rate.aspirate = FLOW_RATE_300ul  # default * 1.3
    right_pipette_300ul.flow_rate.dispense = FLOW_RATE_300ul
    right_pipette_300ul.flow_rate.blow_out = FLOW_RATE_300ul


    # https://docs.opentrons.com/v2/writing.html#commands
    # https://docs.opentrons.com/v2/new_examples.html

    # transfer all the combinations in all the blocks
    for block in BLOCKS:
        # for simplicity, pipettes are provided as a 2-member list with the smaller volume always at index 0 and higher volume at index 1
        # TODO for wider use in other protocols, this should be done in a smarter and more generic way
        block.transfer_block(protocol, [None, right_pipette_300ul], source_plate, target_plate1, VOLUME_PER_TARGET_WELL_UL, CHANGE_TIPS, TOUCH_TIP)


    # test for protocol's logical integrity
    assert sum(required_tip_amounts.values()) == DEBUG_TRANSFER_COUNTER, "pre-calculated total tip usage should equal the amount of executed transfers"

    protocol.comment("Amount of executed transfers at the end is " + str(DEBUG_TRANSFER_COUNTER) + ", total pre-calculated tip consumption was " + str(sum(required_tip_amounts.values())) + ", values are equal: " + str(DEBUG_TRANSFER_COUNTER == sum(required_tip_amounts.values())))

    # DID NOT WORK in separating computer vs robot despite Opentrons Cookbook instructions
    # path: str
    # if not protocol.is_simulating():
    #     path = "/data/user_storage/"
    # else:
    #     path = ""  # to protocol folder

    FileHandler.close_files()

    protocol.comment("Saved transfers data to file " + FileHandler.transfers_filename + " and sources data to file " + FileHandler.sources_filename + " in path " + FileHandler.file_path)






