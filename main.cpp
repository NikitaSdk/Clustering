#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <sstream>


struct Customer {
    int operator_id;
    double call_minutes;
    double sms_count;
    double internet_mb;
    int last_activity;
    int age;
    int prepaid;
};


std::vector<Customer> readDataFromCSV(const std::string& filename) {
    std::vector<Customer> customers;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return customers;
    }

    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Customer c;


        std::getline(ss, value, ',');
        c.operator_id = std::stoi(value);


        std::getline(ss, value, ',');
        c.call_minutes = std::stod(value);


        std::getline(ss, value, ',');
        c.sms_count = std::stod(value);


        std::getline(ss, value, ',');
        c.internet_mb = std::stod(value);


        std::getline(ss, value, ',');
        c.last_activity = std::stoi(value);


        std::getline(ss, value, ',');
        c.age = std::stoi(value);


        std::getline(ss, value, ',');
        c.prepaid = std::stoi(value);

        customers.push_back(c);
    }

    return customers;
}


arma::mat convertToMatrix(const std::vector<Customer>& customers) {
    arma::mat data(7, customers.size());  // 7 features for each customer

    for (size_t i = 0; i < customers.size(); ++i) {
        data(0, i) = customers[i].operator_id;
        data(1, i) = customers[i].call_minutes;
        data(2, i) = customers[i].sms_count;
        data(3, i) = customers[i].internet_mb;
        data(4, i) = customers[i].last_activity;
        data(5, i) = customers[i].age;
        data(6, i) = customers[i].prepaid;
    }

    return data;
}


double euclideanDistance(const arma::vec& a, const arma::vec& b) {
    return arma::norm(a - b);
}


double calculateSilhouetteScore(const arma::mat& data, const arma::Row<size_t>& assignments, int k) {
    double totalScore = 0.0;
    int validPoints = 0;

    for (size_t i = 0; i < data.n_cols; ++i) {

        double a = 0.0;
        int clusterSize = 0;
        for (size_t j = 0; j < data.n_cols; ++j) {
            if (assignments(j) == assignments(i) && i != j) {
                a += euclideanDistance(data.col(i), data.col(j));
                clusterSize++;
            }
        }
        if (clusterSize > 0) {
            a /= clusterSize;
        }


        double b = std::numeric_limits<double>::max();
        for (int cluster = 0; cluster < k; ++cluster) {
            if (cluster != assignments(i)) {
                double clusterDist = 0.0;
                int otherClusterSize = 0;
                for (size_t j = 0; j < data.n_cols; ++j) {
                    if (assignments(j) == cluster) {
                        clusterDist += euclideanDistance(data.col(i), data.col(j));
                        otherClusterSize++;
                    }
                }
                if (otherClusterSize > 0) {
                    clusterDist /= otherClusterSize;
                    b = std::min(b, clusterDist);
                }
            }
        }


        if (b > a) {
            double s = (b - a) / std::max(a, b);
            totalScore += s;
            validPoints++;
        }
    }

    return validPoints > 0 ? totalScore / validPoints : 0.0;
}


double calculateIntraClusterDistance(const arma::mat& data, const arma::Row<size_t>& assignments, int k) {
    double totalDistance = 0.0;
    int totalPoints = 0;

    for (int cluster = 0; cluster < k; ++cluster) {
        std::vector<arma::vec> clusterPoints;
        for (size_t i = 0; i < data.n_cols; ++i) {
            if (assignments(i) == cluster) {
                clusterPoints.push_back(data.col(i));
            }
        }


        for (size_t i = 0; i < clusterPoints.size(); ++i) {
            for (size_t j = i + 1; j < clusterPoints.size(); ++j) {
                totalDistance += euclideanDistance(clusterPoints[i], clusterPoints[j]);
                totalPoints++;
            }
        }
    }

    return totalPoints > 0 ? totalDistance / totalPoints : 0.0;
}


void saveResults(const std::vector<Customer>& customers,
                const arma::Row<size_t>& assignments,
                const arma::mat& centroids,
                const std::string& filename) {
    std::ofstream file(filename);
    file << "operator_id,call_minutes,sms_count,internet_mb,last_activity,age,prepaid,cluster\n";

    for (size_t i = 0; i < customers.size(); ++i) {
        file << customers[i].operator_id << ","
             << customers[i].call_minutes << ","
             << customers[i].sms_count << ","
             << customers[i].internet_mb << ","
             << customers[i].last_activity << ","
             << customers[i].age << ","
             << customers[i].prepaid << ","
             << assignments[i] << "\n";
    }
    file.close();
}

int main() {

    std::vector<Customer> customers = readDataFromCSV("C:\\Users\\user\\CLionProjects\\CodeForTest3WithC\\clients_mobile.csv");

    if (customers.empty()) {
        std::cerr << "No data was read from the file. Exiting " << std::endl;
        return 1;
    }

    std::cout << "Read " << customers.size() << " customers from the file." << std::endl;


    arma::mat data = convertToMatrix(customers);


    arma::mat normalizedData = arma::normalise(data);


    const int k = 3;
    mlpack::KMeans<> kmeans;


    auto start = std::chrono::high_resolution_clock::now();


    arma::Row<size_t> assignments;
    arma::mat centroids;
    kmeans.Cluster(normalizedData, k, assignments, centroids);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double silhouetteScore = calculateSilhouetteScore(normalizedData, assignments, k);
    double intraClusterDistance = calculateIntraClusterDistance(normalizedData, assignments, k);


    std::cout << "Clustering Metrics:\n";
    std::cout << "Execution time: " << duration.count() << " seconds\n";
    std::cout << "Silhouette Score: " << silhouetteScore << "\n";
    std::cout << "Average intra-cluster distance: " << intraClusterDistance << "\n\n";


    std::cout << "Cluster Characteristics:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "\nCluster " << i << ":\n";
        std::cout << "Centroid:\n";
        std::cout << "Operator: " << centroids(0, i) << "\n";
        std::cout << "Call Minutes: " << centroids(1, i) << "\n";
        std::cout << "SMS Count: " << centroids(2, i) << "\n";
        std::cout << "Internet (MB): " << centroids(3, i) << "\n";
        std::cout << "Last Activity (days): " << centroids(4, i) << "\n";
        std::cout << "Age: " << centroids(5, i) << "\n";
        std::cout << "Prepaid: " << centroids(6, i) << "\n";


        int clusterSize = 0;
        for (size_t j = 0; j < assignments.n_elem; ++j) {
            if (assignments(j) == i) clusterSize++;
        }
        std::cout << "Cluster size: " << clusterSize << "\n";


        double avgCalls = 0, avgSms = 0, avgInternet = 0, avgAge = 0;
        int count = 0;
        for (size_t j = 0; j < customers.size(); ++j) {
            if (assignments(j) == i) {
                avgCalls += customers[j].call_minutes;
                avgSms += customers[j].sms_count;
                avgInternet += customers[j].internet_mb;
                avgAge += customers[j].age;
                count++;
            }
        }
        if (count > 0) {
            avgCalls /= count;
            avgSms /= count;
            avgInternet /= count;
            avgAge /= count;
            std::cout << "Average values in cluster:\n";
            std::cout << "Call Minutes: " << avgCalls << "\n";
            std::cout << "SMS Count: " << avgSms << "\n";
            std::cout << "Internet: " << avgInternet << " MB\n";
            std::cout << "Age: " << avgAge << " years\n";
        }
    }


    saveResults(customers, assignments, centroids, "clustering_results.csv");

    return 0;
}
