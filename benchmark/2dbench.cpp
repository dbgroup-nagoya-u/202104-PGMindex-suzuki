/*
 * This example shows how to use pgm::MultidimensionalPGMIndex, a container supporting orthogonal range queries
 * in k dimensions. To run it, your CPU must support the BMI2 instruction set.
 * Compile with:
 *   g++ multidimensional.cpp -std=c++17 -I../include -o multidimensional
 * Run with:
 *   ./multidimensional
 */

#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <getopt.h>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include "pgm/pgm_index_variants.hpp"

int main(int argc, char **argv) {

    int c, cardinality;
    std::string distribution,skeweness = "1";
    static struct option long_options[] =
    {
        {"cardinality" , required_argument , NULL , 'c'},
        {"distribution" , required_argument , NULL , 'd'},
    };
    while(1)
    {
        int opt_index = 0;
        c = getopt_long(argc, argv,"c:d:z", long_options,&opt_index);
        
        if(-1 == c)
        {
            break;
        }
        switch(c)
        {
            case 'c':
                cardinality = atoll(optarg);
                break;
            case 'd':
                distribution = optarg;
                break;
        }
    }

    if(distribution == "skewed"){
        skeweness = "4";
    }else if(distribution == "japan"){
        cardinality = 2030818;
    }else if(distribution == "china"){
        cardinality = 2677695;
    }else{
        cardinality = 1000000;
    }

    std::string filepath = "../dataset/" + distribution + "_" + std::to_string(cardinality) + "_" + skeweness + "_2_.csv";
    std::ifstream ifs(filepath);
    std::string str_buf;

    if (!ifs){
        std::cout << "Error::Can not open " << filepath << std::endl;
        return 0;
    }

    // Generate 2D points from given file
    std::vector<std::tuple<uint64_t, uint64_t>> data;
    while(getline(ifs,str_buf)){
        int i = 0;
        std::string tmp = "";
        std::istringstream stream(str_buf);
        double xy[2];
        while (getline(stream, tmp, ','))
        {
            xy[i] = std::stod(tmp);
            i++;
        }
        std::tuple<uint64_t, uint64_t> coordinate = std::make_tuple(xy[0] * cardinality, xy[1] * cardinality);
        data.push_back(coordinate);
    }
    std::cout << "cardinality : " << data.size() << std::endl;
    ifs.close();
    // Generate 2D window queries
    filepath = "../dataset/query/window/" + distribution + "_" + std::to_string(cardinality) + "_" + skeweness + "_0.000100_1.000000.csv";
    ifs.open(filepath);

    if (!ifs){
        std::cout << "Error::Can not open " << filepath << std::endl;
        return 0;
    }
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>> window_queries;
    while(getline(ifs,str_buf)){
        int i = 0;
        std::string tmp = "";
        std::istringstream stream(str_buf);
        double xy[4];
        while (getline(stream, tmp, ','))
        {
            xy[i] = std::stod(tmp);
            i++;
        }
        std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> window = std::make_tuple(xy[0] * cardinality, xy[1] * cardinality, xy[2] * cardinality, xy[3] * cardinality);
        window_queries.push_back(window);
    }

    // Construct the Multidimensional PGM-index
    constexpr auto dimensions = 2; // number of dimensions
    constexpr auto epsilon = 32;   // space-time trade-off parameter
    auto start = std::chrono::high_resolution_clock::now();

    pgm::MultidimensionalPGMIndex<dimensions, uint64_t, epsilon> pgm_2d(data.begin(), data.end());
    
    auto finish = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    std::cout << "build time , " << time << std::endl;

    // Point qurey for all points in PGM-index
    start = std::chrono::high_resolution_clock::now();
    for(auto point : data){
        pgm_2d.contains({std::get<0>(point) , std::get<1>(point)});
    }
    finish = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0 / data.size();
    std::cout << "point query time , " << time << std::endl;

    // Range query for 1000 times
    start = std::chrono::high_resolution_clock::now();
    for(auto window : window_queries){
        pgm_2d.range({std::get<0>(window) , std::get<1>(window)}, {std::get<2>(window) , std::get<3>(window)});
    }
    finish = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0 / window_queries.size();
    std::cout << "window query time , " << time << std::endl;

    // knn query for 1000 times

    return 0;
}