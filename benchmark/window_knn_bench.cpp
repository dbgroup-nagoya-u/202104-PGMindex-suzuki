/*
 * Benchmark of MultiDimensionalPGM-index using 2-dimensional dataset.
 * 
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

#define QUERY_TIMES 10

std::string path_to_dataset = "../../dataset/";
std::string path_to_query = "../../dataset/query/";

int main(int argc, char **argv) {

    int c, cardinality;
    std::string distribution,skeweness = "1";
    static struct option long_options[] ={
        {"cardinality" , required_argument , NULL , 'c'},
        {"distribution" , required_argument , NULL , 'd'},
    };
    while(1){
        int opt_index = 0;
        c = getopt_long(argc, argv,"c:d:z", long_options,&opt_index);
        
        if(-1 == c) break;

        switch(c){
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
    }else if(distribution == "usa"){
        cardinality = 17383488;
    }else{
        cardinality = 10000000;
    }

    std::string filepath = path_to_dataset + distribution + "_" + std::to_string(cardinality) + "_" + skeweness + "_2_.csv";
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
        while (getline(stream, tmp, ',') and i < 2)
        {
            xy[i] = std::stod(tmp);
            if(xy[i] > 1.0) xy[i] = 1.0;
            else if (xy[i] < 0.0) xy[i] = 0.0;
            i++;
        }
        std::tuple<uint64_t, uint64_t> coordinate = std::make_tuple(xy[0] * cardinality, xy[1] * cardinality);            
        data.push_back(coordinate);
    }
    ifs.close();

    // Generate 2D window queries
    filepath = path_to_query + "window/" + distribution + "_" + std::to_string(cardinality) + "_" + skeweness + "_0.000100_1.000000.csv";
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
    ifs.close();

    // Generate 2D knn queries
    filepath = path_to_query + "knn/" + distribution + "_" + std::to_string(cardinality) + "_" + skeweness + ".csv";
    ifs.open(filepath);

    if (!ifs){
        std::cout << "Error::Can not open " << filepath << std::endl;
        return 0;
    }
    std::vector<std::tuple<uint64_t, uint64_t>> knn_queries;
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
        knn_queries.push_back(coordinate);
    }
    ifs.close();
    
    // Construct the Multidimensional PGM-index
    constexpr auto dimensions = 2; // number of dimensions
    constexpr auto epsilon = 32;   // space-time trade-off parameter


    pgm::MultidimensionalPGMIndex<dimensions, uint64_t, epsilon> pgm_2d(data.begin(), data.end());

    // Range query for 10000 times

    auto start = std::chrono::high_resolution_clock::now();
    auto cnt = 0;

    for(int i = 0; i < QUERY_TIMES; i++){
        for(auto window : window_queries){
            std::vector<std::tuple<uint64_t , uint64_t>> ans;
            for (auto it = pgm_2d.range({std::get<0>(window) , std::get<1>(window)}, {std::get<2>(window) , std::get<3>(window)}); it != pgm_2d.end(); ++it)
                ans.emplace_back(*it);
        }
    }

    
    auto finish = std::chrono::high_resolution_clock::now();
    auto time = (std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0) / (window_queries.size() * QUERY_TIMES);
    std::cout << "window query time , " << time << std::endl;


    // knn query for 10000 times
    std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> ans_knn;
    start = std::chrono::high_resolution_clock::now();
    cnt = 0;

    for(int i = 0;i < QUERY_TIMES; i++){
        for(auto point : knn_queries){
            cnt ++;
            auto ans = pgm_2d.knn({std::get<0>(point) , std::get<1>(point)} , 25);
        }
    }

    finish = std::chrono::high_resolution_clock::now();
    time = (std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0) / (knn_queries.size() * QUERY_TIMES);
    std::cout << "knn query time , " << time << std::endl;

    return 0;
}
