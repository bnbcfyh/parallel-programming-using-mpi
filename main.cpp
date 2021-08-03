#include <iostream>
using namespace std;

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <set>
#include <mpi.h>
#include <stdlib.h>

/*
 * Gets two instances as parameters and returns the
 * manhattan distance between those.
 */
double manhattan_distance(vector<double> &instance1,
		vector<double> &instance2) {
	double result = 0;
	for (int i = 0; i < instance1.size(); ++i)
		result += abs(instance1[i] - instance2[i]);
	return result;
}

/*
 *  Implementation of the diff function given in the Relief algorithm.
 *
 *  Parameters:
 *  feature_index: index number of the spesific feature that we are checking
 * This parameter is used to know which feature to check in the given instances.
 *  target_instance: the target instance in the diff function
 *  hit_or_miss: the nearest hit or the nearest miss
 *  max_min_values: the vector storing maximum and minimum values of the given
 * feature
 *
 *  Returns: the diff calculation according to the formula.
 */
double diff(int feature_index, vector<double> &target_instance,
		vector<double> &hit_or_miss, vector<double> &max_min_values) {

	double value_difference;

	if (hit_or_miss.empty()) { // in case there is no nearest hit/miss
		value_difference = target_instance[feature_index];
	} else {
		value_difference = abs(
				target_instance[feature_index] - hit_or_miss[feature_index]);
	}

	double max_min_difference = max_min_values[1] - max_min_values[0];

	return value_difference / max_min_difference;

}

/*
 * Relief algorithm implementation returning the vector of the weights
 * after running the algorithm "iteration_count" times on the "data" vector.
 */
vector<double> relief(vector<vector<double>> &data, int iteration_count) {

	int instance_count = data.size();
	int feature_count = data[0].size() - 1; // -1 for the class label

	/*
	 * This is a collection of vectors that hold each
	 * feature's maximum and minimum values. Therefore,
	 * it is basically a vector consisting of vectors
	 * of having size 2, holding only maximum and minimum
	 * values of the given feature. Note that the min value
	 * is stored in the first index.
	 */
	vector<vector<double>> max_min_values;

	// find these max and min values
	for (int i = 0; i < feature_count; ++i) {
		double min_value = numeric_limits<double>::max();
		double max_value = numeric_limits<double>::min();
		for (int j = 0; j < instance_count; ++j) {
			if (data[j][i] < min_value)
				min_value = data[j][i];

			if (data[j][i] > max_value)
				max_value = data[j][i];
		}

		vector<double> current_values;
		current_values.push_back(min_value);
		current_values.push_back(max_value);

		max_min_values.push_back(current_values);
	}

	// initialize the weights array
	vector<double> weights;
	for (int i = 0; i < feature_count; ++i) {
		weights.push_back(0);
	}

	// start the iterations
	for (int i = 0; i < iteration_count; ++i) {

		int temp = i;
		i = i % instance_count;

		vector<double> target_instance = data[i]; // instances are picked in order for simplicity
		int target_size = target_instance.size();
		double target_class_label = target_instance[target_size - 1];

		// Find the nearest hit and nearest miss values separately.

		// first calculate the nearest hit
		vector<double> nearest_hit;
		double min_hit_difference = numeric_limits<double>::max();

		for (int j = 0; j < instance_count; ++j) {
			if (i == j) // do not compare it to itself
				continue;

			vector<double> current_instance = data[j];
			int current_size = current_instance.size();
			double current_class_label = current_instance[current_size - 1];

			if (current_class_label != target_class_label) // it is a miss
				continue;

			double difference = manhattan_distance(current_instance,
					target_instance);

			if (difference < min_hit_difference) {
				min_hit_difference = difference;
				nearest_hit = current_instance;
			}

		}

		// then calculate the nearest miss
		vector<double> nearest_miss;
		double min_miss_difference = numeric_limits<double>::max();

		for (int j = 0; j < instance_count; ++j) {
			if (i == j) // do not compare it to itself
				continue;

			vector<double> current_instance = data[j];
			int current_size = current_instance.size();
			double current_class_label = current_instance[current_size - 1];

			if (current_class_label == target_class_label) // it is a hit
				continue;

			double difference = manhattan_distance(current_instance,
					target_instance);

			if (difference < min_miss_difference) {
				min_miss_difference = difference;
				nearest_miss = current_instance;
			}
		}

		// Then, update the weights.

		for (int j = 0; j < feature_count; ++j) {
			weights[j] = weights[j]
					- (diff(j, target_instance, nearest_hit, max_min_values[j])
							/ iteration_count)
					+ (diff(j, target_instance, nearest_miss, max_min_values[j])
							/ iteration_count);
		}
		i = temp;
	}
	return weights;
}

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int processor_count;
	int instance_count;
	int feature_count;
	int iteration_count;
	int top_feature_count;

	vector<vector<double>> data_list;

	if (world_rank == 0) { // master

		string filename = argv[1];
		ifstream input_file(filename.c_str());

		// read the input and hold the values in the variables.
		if (!input_file.is_open()) { // file is not open
			return -1;
		}
		int count = 0;
		string str;

		while (getline(input_file, str)) {
			if (count == 0) { // first line
				processor_count = stoi(str);
				count++;
			} else if (count == 1) { // second line
				istringstream iss(str);
				iss >> instance_count;
				iss >> feature_count;
				iss >> iteration_count;
				iss >> top_feature_count;

				count++;
			} else { // n lines of input
				istringstream iss(str);
				vector<double> numbers;
				double d;
				while (iss >> d)
					numbers.push_back(d);
				data_list.push_back(numbers);
			}
		}

		// send the necessary data to the children

		int slave_count = processor_count - 1;
		int part_size = instance_count / slave_count;

		for (int i = 0; i < slave_count; ++i) {
			int partner_rank = i + 1; // 0th is the master itself

			MPI_Send(&iteration_count, 1, MPI_INT, partner_rank, 0,
					MPI_COMM_WORLD);
			MPI_Send(&top_feature_count, 1, MPI_INT, partner_rank, 0,
					MPI_COMM_WORLD);

			MPI_Send(&feature_count, 1, MPI_INT, partner_rank, 0,
					MPI_COMM_WORLD);

			MPI_Send(&part_size, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);

			for (int j = i * part_size; j < part_size * (i + 1); ++j) {
				for (int k = 0; k < feature_count + 1; ++k) { // add +1 for class label
					double current = data_list[j][k];
					MPI_Send(&current, 1, MPI_DOUBLE, partner_rank, 0,
							MPI_COMM_WORLD);
				}
			}
		}

		// receive the results from the slave processors and process those

		vector<vector<int>> results;
		for (int i = 0; i < slave_count; ++i) {
			int current_result_size;
			MPI_Recv(&current_result_size, 1, MPI_INT, (i + 1), 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			vector<int> current_result;
			for (int j = 0; j < current_result_size; ++j) {
				int number;
				MPI_Recv(&number, 1, MPI_INT, (i + 1), 0, MPI_COMM_WORLD,
						MPI_STATUS_IGNORE);
				current_result.push_back(number);
			}

			results.push_back(current_result);
		}

		cout << "Master P0 : ";

		// create a set without duplicates
		set<int> result_wo_duplicates;
		for (int i = 0; i < results.size(); ++i) {
			for (int j = 0; j < results[i].size(); ++j) {
				result_wo_duplicates.insert(results[i][j]);
			}
		}

		for (set<int>::iterator set_iterator = result_wo_duplicates.begin();
				set_iterator != result_wo_duplicates.end(); ++set_iterator) {
			if (set_iterator == --result_wo_duplicates.end())
				cout << *set_iterator;
			else
				cout << *set_iterator << " ";
		}

	}

	else { //slaves

		MPI_Recv(&iteration_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

		MPI_Recv(&top_feature_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

		MPI_Recv(&feature_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

		int lines_to_read;

		MPI_Recv(&lines_to_read, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

		vector<vector<double>> data;
		for (int i = 0; i < lines_to_read; ++i) {
			vector<double> current_line;

			for (int k = 0; k < feature_count + 1; ++k) { // +1 for the class label
				double current_number;
				MPI_Recv(&current_number, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
						MPI_STATUS_IGNORE);

				current_line.push_back(current_number);
			}

			data.push_back(current_line);
		}

		// get the weights using the relief algorithm
		vector<double> weights = relief(data, iteration_count);

		// copy the array of weights
		vector<double> weights_copy;
		for (int i = 0; i < weights.size(); ++i) {
			weights_copy.push_back(weights[i]);
		}

		// sort the weights array
		sort(weights.begin(), weights.end(), greater<double>());

		// collect the top features in the result array
		vector<int> result;
		for (int i = 0; i < top_feature_count; ++i) {
			double key = weights[i]; // the smallest ith weight
			for (int j = 0; j < weights_copy.size(); ++j) {
				// search the key in the original ordered array to find its index
				double element = weights_copy[j];
				if (element == key) {
					result.push_back(j);
					/*
					 * If there exists another weight having the same value
					 * with the previous one, this will also print the same
					 * instance. To prevent this, we should make sure that
					 * the program does not check the same index again. Since
					 * no weight can be negative, set that index to -1 so that
					 * the program can find the next index having the same
					 * weight value for the next top feature.
					 */
					weights_copy[j] = -1;
					break;
				}
			}
		}

		// sort the results
		sort(result.begin(), result.end());

		// print the results
		cout << "Slave P" + to_string(world_rank) + " : ";
		for (int i = 0; i < result.size() - 1; ++i)
			cout << to_string(result[i]) << " ";

		cout << to_string(result[result.size() - 1]) << endl;

		// send the results to the master
		int size = result.size();
		MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		for (int i = 0; i < result.size(); ++i) {
			int current = result[i];
			MPI_Send(&current, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}

	}

	MPI_Finalize();
	return 0;
}
