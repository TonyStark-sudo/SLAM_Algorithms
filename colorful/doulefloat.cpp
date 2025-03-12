#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using namespace std;

int main(int argc, char *argv[])
{
    nlohmann::json config_ = {
        {"road_hdmapping_bev_z_range", {-1.7424628734588623, 5.4158101081848145}},
        {"road_hdmapping_bev_slack_z", 2.0}
    };
    cout << "road_hdmapping_bev_z_range: " << config_["road_hdmapping_bev_z_range"][1] << endl;
    cout << "road_hdmapping_bev_z_range: " << double(config_["road_hdmapping_bev_z_range"][1]) << endl;
    float translation = float(config_["road_hdmapping_bev_z_range"][1]) + float(config_["road_hdmapping_bev_slack_z"]);
    cout << "float translation: " << translation << endl;
    double translation_d = double(config_["road_hdmapping_bev_z_range"][1]) + double(config_["road_hdmapping_bev_slack_z"]);
    cout << "double translation: " << translation_d << endl;
    /* code */
    return 0;
}
