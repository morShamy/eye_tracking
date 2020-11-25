import numpy as np
from plotnine import *
from numpy import median
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# user control
SUBJECTS_NUMBERS = ["1", "2", "4", "5", "6", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17"]
TESTS_TYPES = ["13pt1sp", "13pt1.2spA", "13pt1.5sp", "13pt2sp", "10pt1.2sp", "13pt1.2spB", "16pt1.2sp", '20pt1.2sp']
THRESHOLD = 0.5

# paths to data
SUMMARIZED_DATA_PATH = "data\\summarized_data.csv"
PATH_BASE_DATA_FORMAT_GAZE = "data\\gaze_fixation_by_subject\\%s_all_gaze.csv"
PATH_BASE_DATA_FORMAT_FIXATIONS = "data\\gaze_fixation_by_subject\\%s_fixations.csv"
PATH_CORDS = "data\\coordinates.csv"

# paths to graphs to save
PATH_TO_SAVE_SCATTER = "graphs\\scatter_with_bias_correction\\%s"
PATH_TO_SAVE_SCATTER_NO_BIAS = "graphs\\scatter_no_bias_correction\\%s"
PATH_TO_SAVE_SCATTER_ONLY_BIAS = "graphs\\scatter_only_bias_correction\\%s"
PATH_TO_SAVE_TIME_UNTIL_QUESTION = "graphs\\time_until_question"
PATH_TO_SAVE_CDF_BY_SUBJECT_NUMBER = "graphs\\cdf\\%s"
PATH_TO_SAVE_SUCCESS_SUMMARY = "graphs\\success_summary"
PATH_TO_COUNTING_SEQUENCES = "graphs\\counting_sequences"
PATH_SAVE_DATA_FILE_WITH_SEQUENCES = "graphs\\jump_to_question_sequences"
FILE_NAME_JUMP_SEQUENCES = "\\jump_to_question_sequences.csv"
PATH_TO_SAVE_COMBINE_CDF = "graphs\\combine_cdf"

# titles of graphs
NUMBER_OF_SUCCESS_TITLE_FORMAT = "successful subjects ratio per test\n %.2f threshold"
BIAS_TITLE = "bias fix is %.2f"
GRAPH_TYPES_DICT = {"SCATTER": 1, "NO BIAS SCATTER": 2, "ONLY BIAS SCATTER": 3, "CDF": 4, "TIME_UNTIL_QUESTION": 5,
                    "SUCCESS_RATE": 6, "COMBINE_CDF": 7, "SEQUENCES": 8}
TITLE_FORMAT_SCATTER = "subject:%s, test: %s, find row:%s, question:%s \n"
TITLE_FORMAT_SCATTER_WITH_BIAS = "subject:%s, test: %s, find row:%s, question:%s\nbias: %f"
TIME_UNTIL_QUESTION_TITLE = "time until reaching question by text appearance order"
TITLE_RATE_OF_SUCCESSFUL_FIXATIONS = "successful fixations ratio per test"
CDF_TITLE = "ECDF of subject fixations\n subject %s, test %s"
COUNTING_SEQUENCES_TITLE = "average number of counting until target row\nby way of finding row"
COMBINE_CDF_TITLE = "ECDF of subject fixations by test type\n subject %s"
TEST_ID_FORMAT = "%s_%s"
X_LABEL = "time(seconds)"
Y_LABEL = "location"
Y_COORDINATE_COLUMN_NAME = "FPOGY"
TIME_COLUMN_NAME = "TIME"
TESTS_TYPES_ORDER_LABEL = ["1\n13pt\n1sp", "2\n13pt\n1.2spA", "3\n13pt\n1.5sp", "4\n13pt\n2sp", "5\n10pt\n1.2sp",
                           "6\n13pt\n1.2spB", "7\n16pt\n1.2sp", '8\n20pt\n1.2sp']
TESTS_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
IGNORE_SAMPLE = 0

# files names
TEST_FILE_NAME = "%s.png"
CDF_SUMMARY_FILE_NAME = "cdf_summary_over_subjects"
TIME_UNTIL_QUESTION_FILE_NAME = "time_until_question"
SUCCESSFUL_FIXATIONS_PER_TEST_FILE_NAME = "successful_fixations_ratio_per_test.png"
SUCCESSFUL_SUBJECTS_RATIO_WITH_THRESHOLD_FILE_NAME = "successful_subjects_ratio_with_threshold.png"
NUMBER_OF_COUNTING_SEQUENCES_FILE_NAME = "number_of_counting_sequences.png"

# Messages to user
OUTPUT_PRINT = 0
OUTPUT_SAVE = 1
GRAPHS_VALID_OPTIONS = "{1, 2, 3, 4, 5, 6, 7, 8, 0}"
OUTPUTS_VALID_OPTIONS = '{' + str(OUTPUT_PRINT) + ',' + str(OUTPUT_SAVE) + '}'

EXIT = '0'
SELECT_GRAPH_MSG = "Please type a graph selection:\n" \
                   "Gazes distribution, with and without bias correction: 1\n" \
                   "Gazes distribution, only with bias correction: 2\n" \
                   "Gazes distribution, without bias correction: 3\n" \
                   "Gazes cdf per subject per test: 4\n" \
                   "Time until reaching question: 5\n" \
                   "Success rate per test: 6\n" \
                   "Gazes cdf after normalization: 7\n" \
                   "Sequences until target and question rows: 8\n" \
                   "Exit program: 0\n" \
                   "your selection: "

SELECT_OUTPUT_MSG = "Please type a form of output: " \
                    "\nprint: " + str(OUTPUT_PRINT) + \
                    "\nsave in local directory: " + str(OUTPUT_SAVE) + \
                    "\nyour selection: "

INVALID_SELECTION_MSG = "selection is invalid, please select one of the following "
SELECT_VALID_GRAPH = INVALID_SELECTION_MSG + str(GRAPHS_VALID_OPTIONS) + \
                     "\nyour selection:"
SELECT_VALID_OUTPUT = INVALID_SELECTION_MSG + str(OUTPUTS_VALID_OPTIONS) + \
                      "\nyour selection:"

# functions
"""save in a directory"""


def save_in_directory(g, file_name, path):
    """
    saves the plot (g), with the given file name in the path.
    :param g: current graph
    :param file_name: name of the file to save
    :param path: path to save the file
    """
    if not os.path.exists(path):
        os.makedirs(path)
    ggsave(g, filename=file_name, path=path, verbose=False)


"""output according to user choice"""


def create_output(g, output_type, file_name, path):
    """
    prints or save the current graph according to user choice
    :param g: current graph
    :param output_type: Print or Save in a directory
    :param file_name: name of the file to save
    :param path: path to save the file
    :return:
    """
    if output_type == OUTPUT_PRINT:
        print(g)
    else:
        save_in_directory(g, file_name, path)


""" scatter plot functions """


def get_row_color(cords, row, row_index):
    """
    :param cords: coordinates of the current text on screen
    :param row: current row in specific text
    :param row_index: index of the current row in the original data
    :return: color for this row in the plot
    """
    row_cord = cords["Y"][row]
    color = "lightgray"
    if row_cord == summarized_data["target_row_up"][row_index] or \
            row_cord == summarized_data["target_row_down"][row_index]:
        color = "blue"
    elif row_cord == summarized_data["question_row_up"][row_index] or \
            row_cord == summarized_data["question_row_down"][row_index]:
        color = "red"
    return color


def scatter_draw_rows(g, row_index, test_type):
    """
    :param g: current plot
    :param row_index: index of the current row in the original data
    :param test_type: current test
    :return: plot g, after drawing horizontal rows in the scatter plot.
    """
    cords = pd.read_csv(PATH_CORDS)
    cords_index_array = np.where(cords["test_type"] == test_type)
    for row in range(cords_index_array[0][0], cords_index_array[0][-1] + 1):
        if "gap" not in cords["row_index"][row]:
            color = get_row_color(cords, row, row_index)
            g = g + geom_hline(yintercept=cords["Y"][row], color=color)
            g = g + geom_hline(yintercept=cords["Y"][row] + cords["height"][row], color=color)
    return g


def scatter_draw_focus_time(g, row_index):
    """
    :param g: current plot
    :param row_index: index of the current row in the original data
    :return: plot g, after drawing focus time vertical lines in the scatter plot.
    """
    g = g + geom_vline(xintercept=summarized_data["focus_start"][row_index], color="black") + \
        geom_vline(xintercept=summarized_data["focus_end"][row_index], color="black")
    return g


def scatter_plot(output_type, bias, row_index, path_subject_fixations, path_subject_gaze, subject_number, test_type):
    """
    :param output_type: Print or Save in a directory
    :param bias: bias calculated for this specific subject
    :param row_index: row_index: index of the current row in the original data
    :param path_subject_fixations: path to the fixations of the specific subject
    :param path_subject_gaze: path to the gazes of the specific subject
    :param subject_number: subject number
    :param test_type: test type
    :return: void, show the scatter plot and saves it.
    """
    df_gaze = get_df(path_subject_gaze, test_type)
    df_fixations = get_df(path_subject_fixations, test_type)
    round_bias = str(round(bias, 4))
    df_gaze["fixed location, bias was: " + round_bias] = df_gaze["original location"] + bias
    df_fixations["fixed location, bias was: " + round_bias] = df_fixations["original location"] + bias
    df_gaze = (pd.melt(df_gaze, id_vars=["TIME"],
                       value_vars=['original location', "fixed location, bias was: " + round_bias],
                       value_name="location"))
    df_fixations = (pd.melt(df_fixations, id_vars=["TIME"],
                            value_vars=['original location', "fixed location, bias was: " + round_bias],
                            value_name="location"))
    g = ggplot()
    g = scatter_draw_rows(g, row_index, test_type)
    g = scatter_draw_focus_time(g, row_index) + xlab(X_LABEL) + ylab(Y_LABEL)
    g = g + ggtitle(TITLE_FORMAT_SCATTER % (subject_number, test_type, summarized_data["find row by"][row_index],
                                            summarized_data["question type"][row_index]))
    g = g + theme(title=element_text(size=10, face='bold'))
    g = g + geom_point(data=df_gaze, mapping=aes(x='TIME', y='location'), fill="lightgray", size=1, stroke=0.05)
    g = g + geom_point(data=df_fixations, mapping=aes(x='TIME', y='location'), fill="black", size=2, stroke=0.25)
    g = g + ylim(round(summarized_data["question_row_down"][row_index], 1) + 0.1, 0) + facet_wrap("variable")
    g = g + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
    create_output(g, output_type, TEST_FILE_NAME % test_type, PATH_TO_SAVE_SCATTER % subject_number)


def scatter_generic_plot(g, df_gaze, df_fixations, row_index):
    """
    generic scatter plot action, for both with bias and without bias graphs
    :param g: current graph
    :param df_gaze: contains all gazes of the current subject in the current test
    :param df_fixations: contains all fixations of the current subject in the current test
    :param row_index: index of the current row in the original datat
    :return: graph after changes
    """
    g = g + geom_point(df_gaze, aes(x='TIME', y='original location'), fill="lightgray", size=1, stroke=0.05)
    g = g + geom_point(df_fixations, aes(x='TIME', y='original location'), fill="black", size=2, stroke=0.25)
    g = g + ylim(round(summarized_data["question_row_down"][row_index], 1) + 0.1, 0)
    g = g + theme(title=element_text(size=10, face="bold"))
    g = g + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
    return g


def scatter_only_bias(output_type, path_subject_gaze, path_subject_fixations, test_type, subject_number, row_index,
                      bias):
    """
    :param output_type: Print or Save in a directory
    :param path_subject_gaze: path to the gazes of the specific subject
    :param path_subject_fixations: path to the fixations of the specific subject
    :param test_type: test type
    :param subject_number: subject number
    :param row_index: index of the current row in the original data
    :param bias: bias calculated for this specific subject
    :return: void - scatter a plot with only the changes after adding the bias to the data
    """
    df_gaze = get_df(path_subject_gaze, test_type)
    df_fixations = get_df(path_subject_fixations, test_type)
    df_gaze['original location'] = df_gaze['original location'] + bias
    df_fixations['original location'] = df_fixations['original location'] + bias
    g = ggplot()
    g = scatter_draw_rows(g, row_index, test_type)
    g = scatter_draw_focus_time(g, row_index) + xlab(X_LABEL) + ylab(Y_LABEL)
    g = g + ggtitle(
        TITLE_FORMAT_SCATTER_WITH_BIAS % (subject_number, test_type, summarized_data["find row by"][row_index],
                                          summarized_data["question type"][row_index], bias))
    g = scatter_generic_plot(g, df_gaze, df_fixations, row_index)
    create_output(g, output_type, TEST_FILE_NAME % test_type, PATH_TO_SAVE_SCATTER_ONLY_BIAS % subject_number)


def scatter_plot_no_bias(output_type, path_subject_gaze, path_subject_fixations, subject_number, test_type, row_index):
    """
    :param output_type: Print or Save in a directory
    :param path_subject_gaze: path to the gazes of the specific subject
    :param path_subject_fixations: path to the fixations of the specific subject
    :param subject_number: subject number
    :param test_type: test type
    :param row_index: index of the current row in the original data
    :return: void - scatter a plot with only the data before adding the bias to the it
    """
    df_gaze = get_df(path_subject_gaze, test_type)
    df_fixations = get_df(path_subject_fixations, test_type)
    g = ggplot()
    g = scatter_draw_rows(g, row_index, test_type)
    g = g + xlab(X_LABEL) + ylab(Y_LABEL) + ggtitle(TITLE_FORMAT_SCATTER % (
        subject_number, test_type, summarized_data["find row by"][row_index],
        summarized_data["question type"][row_index])) + theme(
        title=element_text(size=10, face="bold"))
    g = scatter_draw_focus_time(g, row_index)
    g = scatter_generic_plot(g, df_gaze, df_fixations, row_index)
    create_output(g, output_type, TEST_FILE_NAME % test_type, PATH_TO_SAVE_SCATTER_NO_BIAS % subject_number)


"""" cdf functions """


def cdf_draw_rows(g, row_index, test_type):
    """
    :param g: current plot
    :param row_index: index of the current row in the original data
    :param test_type: current test
    :return: plot g, after drawing horizontal rows in the cdf plot.
    """
    cords = pd.read_csv(PATH_CORDS)
    cords_index_array = np.where(cords["test_type"] == test_type)
    for row in range(cords_index_array[0][0], cords_index_array[0][-1] + 1):
        if "gap" not in cords["row_index"][row]:
            color = get_row_color(cords, row, row_index)
            g = g + geom_vline(xintercept=cords["Y"][row], color=color)
            g = g + geom_vline(xintercept=cords["Y"][row] + cords["height"][row], color=color)
    return g


def ggplot_cdf(output_type, bias, row_index, path_subject_fixations, subject_number, test_type):
    """
    :param output_type: Print or Save in a directory
    :param bias: bias calculated for this specific subject
    :param row_index: index of the current row in the original data
    :param path_subject_fixations: path to the fixations of the specific subject
    :param subject_number: subject number
    :param test_type: test type
    :return: void, plots fixations locations of a subject during focus time of a specific test.
    """
    df_fixations = get_df(path_subject_fixations, test_type)
    df_fixations['original location'] = df_fixations['original location'] + bias
    start_focus = summarized_data["focus_start"][row_index]
    end_focus = summarized_data["focus_end"][row_index]
    df_fixations = extract_focus_time(df_fixations, start_focus, end_focus)
    g = ggplot(aes("original location"))
    g = cdf_draw_rows(g, row_index, test_type)
    g = g + ggtitle(CDF_TITLE % (subject_number, test_type))
    g = g + stat_ecdf(data=df_fixations, geom="line", size=1)
    g = g + labs(x="fixation location", y="ECDF")
    g = generic_ggplot(g)
    create_output(g, output_type, TEST_FILE_NAME % test_type, PATH_TO_SAVE_CDF_BY_SUBJECT_NUMBER % subject_number)


""" combine cdf """


# create

def get_target_row_location(row_index):
    """
    :param row_index: specific row index
    :return: location of the target row
    """
    target_row_up = summarized_data["target_row_up"][row_index]
    target_row_down = summarized_data["target_row_down"][row_index]
    return (target_row_up + target_row_down) / 2


def combine_cdf(path_subject_fixations, test_type, bias, row_index):
    """
    :param path_subject_fixations: path to subject fixations data frame
    :param test_type: type of the specific test
    :param bias: current bias
    :param row_index: specific row index
    :return: data frame of fixations after adding bias and more changes
    """
    # get the target row location
    target_row_location = get_target_row_location(row_index)

    # normalize fixations location and add the subject bias
    df_fixations = get_df(path_subject_fixations, test_type)
    df_fixations["test type"] = test_type
    df_fixations['original location'] = df_fixations['original location'] + bias

    # plot
    start_focus = summarized_data["focus_start"][row_index]
    end_focus = summarized_data["focus_end"][row_index]
    df_fixations = extract_focus_time(df_fixations, start_focus, end_focus)
    df_fixations = extract_jump_to_question(df_fixations, test_type, row_index)
    df_fixations = df_fixations[["test type", "original location"]]

    df_fixations['original location'] = df_fixations['original location'] - target_row_location

    return df_fixations


def ggplot_combine_cdf(output_type, df_fixations, subject_number):
    """
    :param output_type: Print or Save in a directory
    :param df_fixations: subject fixations data frame
    :param subject_number: current subject number (from 1-17)
    :return: void, plots the cdf graph
    """
    g = ggplot(aes("original location", color="test type"))
    g = g + theme(legend_position=(0.9, 0.4))
    g = g + ggtitle(COMBINE_CDF_TITLE % subject_number)
    g = g + stat_ecdf(data=df_fixations, geom="line", size=1)
    g = g + labs(x="fixation location", y="ECDF")
    g = generic_ggplot(g)
    create_output(g, output_type, TEST_FILE_NAME % subject_number, PATH_TO_SAVE_COMBINE_CDF)


def ggplot_combine_cdf_summary(output_type, df_fixations):
    """
    :param output_type: Print or Save in a directory
    :param df_fixations: subject fixations data frame
    :return: void, plots the combine cdf graph
    """

    g = ggplot(aes("original location", color="test type"))
    g = g + theme(legend_position=(0.9, 0.4))
    g = g + stat_ecdf(data=df_fixations, geom="line", size=1)
    g = g + labs(x="fixation location", y="ECDF")
    g = generic_ggplot(g)
    g = g + xlim(-0.1, 0.2)
    g = g + scale_color_discrete(breaks=TESTS_TYPES)
    create_output(g, output_type, CDF_SUMMARY_FILE_NAME, PATH_TO_SAVE_COMBINE_CDF)


""" time until question """


def ggplot_time_until_question(output_type):
    """
    :param output_type: Print or Save in a directory
    :return: void, plots the graph
    """
    g = ggplot(summarized_data)
    g = g + geom_point(mapping=aes(x="text_order", y="question_start", group="subject_number"), color="lightgray")
    g = g + geom_line(mapping=aes(x="text_order", y="question_start", group="subject_number"), color="lightgray")
    g = g + xlab("order of text appearance") + ylab("time(seconds)")
    g = g + ggtitle(TIME_UNTIL_QUESTION_TITLE)
    g = g + scale_x_continuous(breaks=TESTS_ORDER)
    g = generic_ggplot(g)
    median_by_order = []
    for order in TESTS_ORDER:
        median_by_order.append(median(summarized_data[summarized_data["text_order"] == order]["question_start"]))
    d = {'x': TESTS_ORDER, 'y': median_by_order}
    df = pd.DataFrame(d)
    g = g + geom_line(data=df, mapping=aes(x='x', y='y')) + geom_point(data=df, mapping=aes(x='x', y='y'))
    create_output(g, output_type, TIME_UNTIL_QUESTION_FILE_NAME, PATH_TO_SAVE_TIME_UNTIL_QUESTION)


""" success functions"""


def ggplot_successful_fixations_ratio_per_test(output_type):
    """
    :return: void, plots rate of successful fixations among all subjects for each test
    """
    success_rate = []
    for test_type in TESTS_TYPES:
        only_test_type = (summarized_data[summarized_data["test_type"] == test_type])
        total_successful_fixations = only_test_type["successes number"].sum()
        total_fixations_without_jump_to_question = (
                only_test_type["total fixations"] - only_test_type["question fixations"]).sum()
        success_rate.append(total_successful_fixations / total_fixations_without_jump_to_question)

    success_df = pd.DataFrame()
    success_df["test_type"] = TESTS_TYPES
    success_df["success_rate"] = success_rate
    success_df["order_test_type"] = TESTS_TYPES_ORDER_LABEL
    success_rate = np.round(success_rate, 2)

    g = ggplot(success_df, aes(x="order_test_type", y="success_rate")) + geom_col(color="black", fill="lightblue")
    g = g + geom_text(label=success_rate, angle=0, size=12, va="baseline")
    g = generic_ggplot(g)
    g = g + xlab("test type") + ylab("successful fixations ratio") + ylim(0, 1)
    g = g + ggtitle(TITLE_RATE_OF_SUCCESSFUL_FIXATIONS)
    create_output(g, output_type, SUCCESSFUL_FIXATIONS_PER_TEST_FILE_NAME, PATH_TO_SAVE_SUCCESS_SUMMARY)


def total_tests_preformed_vector(test_data):
    """
    :param test_data: original data frame with data on each test of each subject
    :return: number of times each test was preformed
    """
    total_test_vector = []
    for index in range(len(TESTS_TYPES)):
        total_test_vector.append((test_data["test_type"] == TESTS_TYPES[index]).sum())
    return total_test_vector


def find_gap_question_start(test_type):
    """
    :param test_type: test type
    :return: coordinate of the start of gap question (end of last row).
    """

    cords = pd.read_csv(PATH_CORDS)
    cords = cords[cords["test_type"] == test_type]
    cords = cords[cords["row_index"] == "gapQuestion"]
    gapQuestion_Start = cords["Y"]
    return float(gapQuestion_Start)


def extract_jump_to_question(df_fixations_focus, test_type, row_index):
    """
    :param df_fixations_focus: data frame of fixations of specific subject in
    :param test_type: type of the test preformed
    :param row_index:  index of the current row in the original data
    :return: cut of the data frame - specific test of a specific subject during the focus time.
    """
    gapQuestion_Start = find_gap_question_start(test_type)
    df_fixations_focus_no_question = df_fixations_focus.loc[df_fixations_focus["original location"] < gapQuestion_Start]
    total_fixations = len(df_fixations_focus)
    summarized_data.loc[row_index, "total fixations"] = total_fixations
    question_fixations = total_fixations - len(df_fixations_focus_no_question)
    summarized_data.loc[row_index, "question fixations"] = question_fixations
    if total_fixations != 0:
        summarized_data.loc[row_index, "question fixations rate"] = question_fixations / total_fixations
    else:
        summarized_data.loc[row_index, "question fixations rate"] = 0
    return df_fixations_focus_no_question


def calculate_target_row_limits(test_type, row_index):
    """
    :param test_type: type of the test preformed
    :param row_index:  index of the current row in the original data
    :return: given a target row, calculates upper and bottom limits of the row.
    """
    cords = pd.read_csv(PATH_CORDS)
    cords = cords[cords["test_type"] == test_type]
    cords = cords[cords["row_index"] == "gap0"]
    gap_half_height = cords["height"] / 2
    target_row_up = summarized_data["target_row_up"][row_index]
    target_row_down = summarized_data["target_row_down"][row_index]
    return float(target_row_up - gap_half_height), float(target_row_down + gap_half_height)


def calculate_success_rate(df_fixation, success_upper_limit, success_bottom_limit):
    """
    :param df_fixation: fixations of the specific subject
    :param success_upper_limit: upper limit of the target row
    :param success_bottom_limit: bottom limit of the target row
    :return: success rate for the specific subject in the specific test
    """
    number_of_fixations = len(df_fixation)
    upper_condition = success_upper_limit < df_fixation["original location"]
    bottom_condition = success_bottom_limit > df_fixation["original location"]
    number_of_success = np.logical_and(upper_condition, bottom_condition).sum()
    if number_of_fixations == 0:
        return 0
    else:
        return number_of_success, (number_of_success / number_of_fixations)


def find_subject_success_in_test(path_subject_fixations, row_index, test_type, bias):
    """
    :param path_subject_fixations:
    :param row_index:  index of the current row in the original data
    :param test_type: test type
    :param bias: bias calculated for this specific subject
    :return: void, updates the original data table with success rate of the subject in the test.
    """
    # take df and add bias
    df_fixations = get_df(path_subject_fixations, test_type)
    df_fixations["original location"] = df_fixations["original location"] + bias
    # cut to just focus time
    start_focus = summarized_data["focus_start"][row_index]
    end_focus = summarized_data["focus_end"][row_index]
    df_fixations_focus = extract_focus_time(df_fixations, start_focus, end_focus)
    # cut off "jump to questions" fixations
    add_number_of_question_sequence(df_fixations_focus, test_type, row_index)
    df_fixations_focus = extract_jump_to_question(df_fixations_focus, test_type, row_index)

    # calculate limits of the success area
    success_upper_limit, success_bottom_limit = calculate_target_row_limits(test_type, row_index)
    number_of_successes, success_rate = calculate_success_rate(df_fixations_focus, success_upper_limit,
                                                               success_bottom_limit)
    # add to the column
    summarized_data.loc[row_index, "successes number"] = number_of_successes
    summarized_data.loc[row_index, "success rate"] = success_rate
    summarized_data.loc[row_index, "is success"] = success_rate > THRESHOLD


def ggplot_successful_subjects_ratio_with_threshold(output_type):
    """
    :return: void, plots a success rate bar plot, for each test - percent of successful subjects.
    """
    num_of_success = []
    for test_type in TESTS_TYPES:
        only_test_type = (summarized_data[summarized_data["test_type"] == test_type])
        num_of_success.append(len(only_test_type[only_test_type["is success"]]))

    success_df = pd.DataFrame()
    success_df["test_type"] = TESTS_TYPES
    success_df["num_of_success"] = num_of_success
    success_rate_vector = np.round(np.divide(num_of_success, total_tests_preformed_vector(summarized_data)), 2)
    success_df["success_rate"] = success_rate_vector
    success_df["order_test_type"] = TESTS_TYPES_ORDER_LABEL

    g = ggplot(success_df, aes(x="order_test_type", y="success_rate")) + geom_col(color="black", fill="lightblue")
    g = g + geom_text(label=success_rate_vector, angle=0, size=12, va="baseline")
    g = generic_ggplot(g)
    g = g + xlab("test type") + ylab("successful subjects ratio") + ylim(0, 1)
    g = g + ggtitle(NUMBER_OF_SUCCESS_TITLE_FORMAT % THRESHOLD)
    create_output(g, output_type, SUCCESSFUL_SUBJECTS_RATIO_WITH_THRESHOLD_FILE_NAME, PATH_TO_SAVE_SUCCESS_SUMMARY)


""" sequences find functions"""


def add_number_of_question_sequence(df_fixations_focus, test_type, row_index):
    """
    adds number of questions sequences to the data frame
    :param df_fixations_focus:
    :param test_type:
    :param row_index:
    :return:
    """
    number_of_jumps = 0
    cur_on_a_sequence = False
    last_row_bottom_location = find_gap_question_start(test_type)
    bool_vector = df_fixations_focus["original location"] >= last_row_bottom_location
    for item in bool_vector:
        if cur_on_a_sequence:
            if not item:
                cur_on_a_sequence = False
        else:
            if item:
                number_of_jumps += 1
                cur_on_a_sequence = True
    summarized_data.loc[row_index, "num of question sequences"] = number_of_jumps


def jump_to_question_sequences(output_type, cut_data):
    """
    :param output_type: Print or Save in a directory
    :param cut_data: part of the original data frame, with only relevant parts
    :return: void, plots bar plot that shows average number of jumping to question sequences by type of question.
    """
    df_exist = cut_data[cut_data["question type"] == "exist"]
    df_count = cut_data[cut_data["question type"] == "count"]
    number_of_exist_question = len(df_exist)
    number_of_count_question = len(df_count)

    average_exist = np.mean(df_exist["num of question sequences"])
    average_count = np.mean(df_count["num of question sequences"])
    mean_sequences_vector = [average_exist, average_count]

    se = [np.std(df_exist["num of question sequences"]) / np.sqrt(number_of_exist_question),
          np.std(df_count["num of question sequences"]) / np.sqrt(number_of_count_question)]

    question_types_vector = ["exist", "count"]
    df = pd.DataFrame()
    df["type"] = question_types_vector
    df["samples number"] = [number_of_exist_question, number_of_count_question]
    df["mean"] = mean_sequences_vector
    df["se"] = se

    if output_type == OUTPUT_PRINT:
        print(df)

    else:
        if not os.path.exists(PATH_SAVE_DATA_FILE_WITH_SEQUENCES):
            os.makedirs(PATH_SAVE_DATA_FILE_WITH_SEQUENCES)
        df.to_csv(PATH_SAVE_DATA_FILE_WITH_SEQUENCES + FILE_NAME_JUMP_SEQUENCES)


def bar_plot_count_until_target(output_type, cut_data):
    """
    :param output_type: Print or Save in a directory
    :param cut_data: part of the original data frame, with only relevant parts
    :return: void, plots bar plot that shows average number of rows counting sequences until target row, by type
     of directions to the question
    """

    # extract only data that is directed to the seventh row by number
    seventh_row_first_index = cut_data["test_type"] == "20pt1.2sp"
    seventh_row_second_index = cut_data["test_type"] == "13pt2sp"
    seventh_row_vector = np.logical_or(seventh_row_first_index, seventh_row_second_index)
    count_data = cut_data[seventh_row_vector]
    indented_data = cut_data[cut_data["find row by"] == "indented"]
    bold_data = cut_data[cut_data["find row by"] == "bold"]
    number_of_count_row_questions = len(count_data)
    number_of_indented_questions = len(indented_data)
    number_of_bold_questions = len(bold_data)

    mean_count = np.mean(count_data["num of row counting"])
    mean_indented = np.mean(indented_data["num of row counting"])
    mean_bold = np.mean(bold_data["num of row counting"])
    mean = [mean_count, mean_indented, mean_bold]

    se_count = np.std(count_data["num of row counting"]) / np.sqrt(number_of_count_row_questions)
    se_indented = np.std(indented_data["num of row counting"]) / np.sqrt(number_of_indented_questions)
    se_bold = np.std(bold_data["num of row counting"]) / np.sqrt(number_of_bold_questions)
    se = [se_count, se_indented, se_bold]

    # create data frame
    df = pd.DataFrame()
    find_row_by_vector = ["number", "indented", "bold"]
    samples_number = [number_of_count_row_questions, number_of_indented_questions, number_of_bold_questions]
    mean = np.round(mean, 2)
    df["find row by"] = find_row_by_vector
    df["samples number"] = samples_number
    df["average counting"] = mean
    df["se"] = se

    # plot
    g = ggplot(df, aes(x="find row by", y="average counting")) + geom_col(color="black", fill="lightblue")
    g = g + geom_text(label=mean, angle=0, size=12, va="baseline")
    g = g + geom_errorbar(aes(ymin=mean - se, ymax=mean + se))
    g = generic_ggplot(g)
    g = g + xlab("find row by") + ylab("average number of rows counting sequences") + ylim(0, 2)
    g = g + ggtitle(COUNTING_SEQUENCES_TITLE)
    create_output(g, output_type, NUMBER_OF_COUNTING_SEQUENCES_FILE_NAME, PATH_TO_COUNTING_SEQUENCES)


""" generic functions """


def generic_ggplot(g):
    """
    :param g: plot
    :return: plot after adding some generic elemnt, that are basic for each graph created in the program
    """
    g = g + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
    g = g + theme(panel_background=element_blank())
    g = g + theme(axis_line=element_line("black"))
    g = g + theme(title=element_text(size=10, face="bold"))

    return g


def get_df(path, type_of_test):
    """
    :param path: path to a data frame
    :param type_of_test: type of the specific test
    :return: data frame after changing some of it's columns and dropping some unnecessary rows
    """
    df = pd.read_csv(path)
    df.rename(columns={df.columns[1]: TIME_COLUMN_NAME}, inplace=True)
    df = df.loc[df["MEDIA_NAME"] == type_of_test]
    df = df.loc[df["FPOGV"] == 1]
    df = df[[Y_COORDINATE_COLUMN_NAME, TIME_COLUMN_NAME]]
    df = df.rename(columns={"FPOGY": "original location"})
    return df


def extract_focus_time(df, start_focus, end_focus):
    """
    :param df: data frame of gazes or fixations of a specific subject in a specific test
    :param start_focus: start of the focus time of a specific subject in a specific test
    :param end_focus: end of the focus time of a specific subject in a specific test
    :return: only rows in the data frame that their time is in the range of the start and end time of the focus tims
    """
    df = df.loc[start_focus < df["TIME"]]
    df = df.loc[df["TIME"] < end_focus]
    return df


def calculate_bias_in_a_test(target_row_coordinate, df):
    """
    :param target_row_coordinate: coordinate of the target row, in the specific test
    :param df: data frame contains gazes or fixations of the specific subject in the specific test
    :return: bias in the specific test of a specific subject
    """
    median_gaze_bias = np.median(df["original location"])
    return target_row_coordinate - median_gaze_bias


def find_bias(subject_number):
    """
    :param subject_number: subject number
    :return: bias of the specific subject
    """
    bias_gaze_array = []
    for test in TESTS_TYPES:
        test_id = TEST_ID_FORMAT % (subject_number, test)
        row_index_array = np.where(summarized_data["test_id"] == test_id)
        if len(row_index_array[0]) == 1:
            row_index = row_index_array[0][0]
            path_subject_gaze = PATH_BASE_DATA_FORMAT_GAZE % subject_number
            target_row_coordinate = (summarized_data["target_row_up"][row_index] + summarized_data["target_row_down"][
                row_index]) / 2
            start_focus = summarized_data["focus_start"][row_index]
            end_focus = summarized_data["focus_end"][row_index]
            df_gaze = extract_focus_time(get_df(path_subject_gaze, test), start_focus, end_focus)
            test_gaze_bias = calculate_bias_in_a_test(target_row_coordinate, df_gaze)
            bias_gaze_array.append(test_gaze_bias)
    bias_gaze = np.mean(bias_gaze_array)
    return bias_gaze


def read_summarized_data_file():
    """
    reads the main data file and makes some changes in it
    """
    df = pd.read_csv(SUMMARIZED_DATA_PATH)
    test_id = (df["subject_number"].astype(str)) + "_" + df["test_type"]
    df["test_id"] = test_id
    return df


""" main functions """


def execute_program(graph_type, output_type):
    """
    :param output_type: Print or Save in a directory
    :param graph_type: specific graph to plot or data to create, by user selection.
    :return: void, execute a specific program between some options.
    """
    if graph_type == GRAPH_TYPES_DICT["TIME_UNTIL_QUESTION"]:
        ggplot_time_until_question(output_type)
    else:
        df_combine_cdf_summary = pd.DataFrame()
        for subject_number in SUBJECTS_NUMBERS:
            bias = find_bias(subject_number)
            df_combine_cdf = pd.DataFrame()
            for test_type in TESTS_TYPES:
                test_id = TEST_ID_FORMAT % (subject_number, test_type)
                row_index_array = np.where(summarized_data["test_id"] == test_id)
                if len(row_index_array[0]) == 1:
                    row_index = row_index_array[0][0]
                    path_subject_fixations = PATH_BASE_DATA_FORMAT_FIXATIONS % subject_number
                    path_subject_gaze = PATH_BASE_DATA_FORMAT_GAZE % subject_number
                    if graph_type == GRAPH_TYPES_DICT["SCATTER"]:
                        scatter_plot(output_type, bias, row_index, path_subject_fixations, path_subject_gaze,
                                     subject_number, test_type)
                    elif graph_type == GRAPH_TYPES_DICT["NO BIAS SCATTER"]:
                        scatter_plot_no_bias(output_type, path_subject_gaze, path_subject_fixations, subject_number,
                                             test_type, row_index)
                    elif graph_type == GRAPH_TYPES_DICT["ONLY BIAS SCATTER"]:
                        scatter_only_bias(output_type, path_subject_gaze, path_subject_fixations, test_type,
                                          subject_number, row_index, bias)
                    elif graph_type == GRAPH_TYPES_DICT["CDF"]:
                        ggplot_cdf(output_type, bias, row_index, path_subject_fixations, subject_number, test_type)
                    elif graph_type == GRAPH_TYPES_DICT["COMBINE_CDF"]:
                        df_combine_cdf = pd.concat([df_combine_cdf, combine_cdf(path_subject_fixations, test_type, bias,
                                                                                row_index)])
                    elif graph_type == GRAPH_TYPES_DICT["SUCCESS_RATE"] or graph_type == GRAPH_TYPES_DICT["SEQUENCES"]:
                        find_subject_success_in_test(path_subject_fixations, row_index, test_type, bias)
            if graph_type == GRAPH_TYPES_DICT["COMBINE_CDF"]:
                ggplot_combine_cdf(output_type, df_combine_cdf, subject_number)
                df_combine_cdf_summary = pd.concat([df_combine_cdf_summary, df_combine_cdf])

        if graph_type == GRAPH_TYPES_DICT["SUCCESS_RATE"]:
            ggplot_successful_fixations_ratio_per_test(output_type)
            ggplot_successful_subjects_ratio_with_threshold(output_type)

        if graph_type == GRAPH_TYPES_DICT["SEQUENCES"]:
            sequences_analysis_data = summarized_data[summarized_data["ignore_sample"] == IGNORE_SAMPLE]
            bar_plot_count_until_target(output_type, sequences_analysis_data)
            jump_to_question_sequences(output_type, sequences_analysis_data)

        if graph_type == GRAPH_TYPES_DICT["COMBINE_CDF"]:
            ggplot_combine_cdf_summary(output_type, df_combine_cdf_summary)


if __name__ == '__main__':
    selection = input(SELECT_GRAPH_MSG)

    while selection != EXIT:
        summarized_data = read_summarized_data_file()

        while selection not in GRAPHS_VALID_OPTIONS:
            selection = input(SELECT_VALID_GRAPH)

        output = input(SELECT_OUTPUT_MSG)
        while output not in OUTPUTS_VALID_OPTIONS:
            output = input(SELECT_VALID_OUTPUT)

        execute_program(int(selection), int(output))

        selection = input(SELECT_GRAPH_MSG)
