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

//#define recall

struct mbr{
    uint64_t x1;
    uint64_t y1;
    uint64_t x2;
    uint64_t y2;
};

std::vector<std::tuple<uint64_t, uint64_t>> acc_window(std::tuple<uint64_t , uint64_t , uint64_t , uint64_t> &window, std::vector<std::tuple<uint64_t, uint64_t>> &data);
std::vector<std::tuple<uint64_t, uint64_t>> acc_knn(std::tuple<uint64_t , uint64_t> query_point, std::vector<std::tuple<uint64_t, uint64_t>> &data , int k);
bool contains(std::tuple<uint64_t , uint64_t , uint64_t , uint64_t> window, std::tuple<uint64_t, uint64_t> point);
double calc_recall(std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> &acc , std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> &pred);

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
    }else{
        cardinality = 10000000;
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
    ifs.close();

    // Generate 2D knn queries
    filepath = "../dataset/query/knn/" + distribution + "_" + std::to_string(cardinality) + "_" + skeweness + ".csv";
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
    int ks[] = {1, 5, 25, 125, 625};

    for (int k : ks){
        std::cout << "k : " << k << " -------- " << std::endl; 
        // knn query for 1000 times
        std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> ans_knn;
        auto start = std::chrono::high_resolution_clock::now();
        for(auto point : knn_queries){
            //std::vector<std::tuple<uint64_t , uint64_t>> ans;
            auto ans = pgm_2d.knn({std::get<0>(point) , std::get<1>(point)} , k);
            ans_knn.push_back(ans);
            //std::cout << ans_knn.size() << std::endl;
        }
        auto finish = std::chrono::high_resolution_clock::now();
        auto dtime = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0 / knn_queries.size();
        std::cout << "knn query time , " << dtime << std::endl;

        // proposed knn query for 1000 times
        std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> ans_knn_proposed;
        start = std::chrono::high_resolution_clock::now();
        for(auto point : knn_queries){
            //std::vector<std::tuple<uint64_t , uint64_t>> ans;
            auto ans = pgm_2d.knn_proposed({std::get<0>(point) , std::get<1>(point)} , k);
            ans_knn_proposed.push_back(ans);
            //std::cout << ans.size() << std::endl;
        }
        finish = std::chrono::high_resolution_clock::now();
        dtime = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0 / knn_queries.size();
        std::cout << "proposed knn query time , " << dtime << std::endl;

        // ACC knn query for 1000 times
        #ifdef recall 
        std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> acc_ans_knn;
        for(auto point : knn_queries){
            std::vector<std::tuple<uint64_t , uint64_t>> ans;
            ans = acc_knn(point , data , 25);
            acc_ans_knn.push_back(ans);
        }
        std::cout << "knn query recall , " << calc_recall(acc_ans_knn , ans_knn) << std::endl;
        std::cout << "proposed knn query recall , " << calc_recall(acc_ans_knn_proposed , ans_knn) << std::endl;
        #endif
    }
    return 0;
}

std::vector<std::tuple<uint64_t, uint64_t>> acc_window(std::tuple<uint64_t , uint64_t , uint64_t , uint64_t> &window, std::vector<std::tuple<uint64_t, uint64_t>> &data){
    std::vector<std::tuple<uint64_t, uint64_t>> ans;
    for (auto point : data){
        if(contains(window , point)) ans.push_back(point);
    }
    return ans;
}

bool contains(std::tuple<uint64_t , uint64_t , uint64_t , uint64_t> window, std::tuple<uint64_t, uint64_t> point){
    return (std::get<0>(window) <= std::get<0>(point) && std::get<0>(point) <= std::get<2>(window) && std::get<1>(window) <= std::get<1>(point) && std::get<1>(point) <= std::get<3>(window));
}

double calc_recall(std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> &acc , std::vector<std::vector<std::tuple<uint64_t, uint64_t>>> &pred){
    int all = 0;
    double sum = 0.0;
    double product = 1.0;
    int all_acc = acc.size();
    for(int i = 0 ; i < acc.size() ; i++){
        all += acc[i].size();
        int acc_size = acc[i].size();
        int collect_size = 0;
        std::map<uint64_t, std::vector<uint64_t>> ans;
        for(auto acc_point : acc[i]){   
            uint64_t ans_x = std::get<0>(acc_point);
            uint64_t ans_y = std::get<1>(acc_point);
            if(ans.count(ans_x) == 0) ans.insert(std::pair<uint64_t,std::vector<uint64_t>> (ans_x, std::vector<uint64_t>()));
            ans[ans_x].push_back(ans_y);
        }
        for(auto pred_point : pred[i]){
            uint64_t pred_x = std::get<0>(pred_point);
            uint64_t pred_y = std::get<1>(pred_point);
            if (std::find(ans[pred_x].begin() , ans[pred_x].end(), pred_y) != ans[pred_x].end()) collect_size += 1;
        }
        if(acc_size == 0){
            all_acc -= 1;
            continue;
        }
        double accuracy = collect_size * 1.0 / acc_size;
        sum += accuracy;
        product *= accuracy;
    }
    return sum * 1.0 / all_acc;
}

std::vector<std::tuple<uint64_t, uint64_t>> acc_knn(std::tuple<uint64_t , uint64_t> query_point, std::vector<std::tuple<uint64_t, uint64_t>> &data , int k){
    auto dist = [&](std::tuple<uint64_t , uint64_t> point){
        return std::sqrt(std::pow(std::abs(int64_t(std::get<0>(query_point) - std::get<0>(point))) , 2) + std::pow(std::abs(int64_t(std::get<1>(query_point) - std::get<1>(point))) , 2));
    };
    std::sort(data.begin() , data.end() , [&](auto const& lhs, auto const& rhs) {
        double dist_l = dist(lhs);
        double dist_r = dist(rhs);
        return dist_l < dist_r;
    });
    std::vector<std::tuple<uint64_t, uint64_t>> ans{data.begin(), data.begin() + k};
    return ans;
}