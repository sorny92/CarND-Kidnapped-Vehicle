/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Random number generator
    default_random_engine generator;
    num_particles = 1000;
    weights.resize(num_particles);

    //Normal distribution generation
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for(int i = 0; i < num_particles; ++i){
        double particle_x, particle_y, particle_theta;
        Particle particle;
        particle.x = dist_x(generator);
        particle.y = dist_y(generator);
        particle.theta = dist_theta(generator);
        particle.weight = 1;
        particles.push_back(particle);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine generator;
    //Normal distribution generation
    for (auto& particle:  particles){
        const double theta = particle.theta + (yaw_rate*delta_t);
        const double x = particle.x + (velocity/yaw_rate)*(sin(theta) - sin(particle.theta));
        const double y = particle.y + (velocity/yaw_rate)*(cos(particle.theta) - cos(theta));

        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);

        particle.x = dist_x(generator);
        particle.y = dist_y(generator);
        particle.theta = dist_theta(generator);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    for(int i = 0; i < observations.size(); ++i){
        double min_distance = 1000;
        int closest_landmark = -1;
        for(int j = 0; j < predicted.size(); ++j){
            double distance = sqrt((predicted[j].x-observations[i].x)*(predicted[j].x-observations[i].x) + 
                            (predicted[j].y-observations[i].y)*(predicted[j].y-observations[i].y));
            if (distance < min_distance) {
                min_distance = distance;
                closest_landmark = j;
            }
        }
        observations[i].id = closest_landmark;
        if(observations[i].id == -1){
            cout << "WHAAAAT" << endl;
        }
        //cout << predicted[j].id << endl;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double total_weight = 0;
    for (int i = 0; i < particles.size(); i++) {
        std::vector<LandmarkObs> car_ref_observations;
        for (int j = 0; j < observations.size(); ++j){
            LandmarkObs landmark_aux;
            landmark_aux.x = observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta) + particles[i].x;
            landmark_aux.y = observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta) + particles[i].y;
            landmark_aux.id = -1;
            car_ref_observations.push_back(landmark_aux);
        }
        vector<LandmarkObs> landmarks_in_range;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
            LandmarkObs landmark = {
                map_landmarks.landmark_list[j].id,
                map_landmarks.landmark_list[j].x,
                map_landmarks.landmark_list[j].y,
            };
            const double distance = sqrt(pow(landmark.x - particles[i].x,2) + 
                                         pow(landmark.y - particles[i].y,2));
            if (distance <= sensor_range){
                landmarks_in_range.push_back(landmark);
            }
        }
        if (landmarks_in_range.size() == 0){
            for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
                LandmarkObs landmark = {
                    map_landmarks.landmark_list[j].id,
                    map_landmarks.landmark_list[j].x,
                    map_landmarks.landmark_list[j].y,
                };
                const double distance = sqrt(pow(landmark.x - particles[i].x,2) + 
                                             pow(landmark.y - particles[i].y,2));
                if (distance <= sensor_range){
                    landmarks_in_range.push_back(landmark);
                }
            }
        }

        dataAssociation(landmarks_in_range, car_ref_observations);


        for (auto& observation : car_ref_observations){
            //cout << "id " << observation.id << endl;
            LandmarkObs landmark = landmarks_in_range[observation.id];

            double gauss_norm = (1/(2*M_PI*std_landmark[0]*std_landmark[1]));
            double exponent = (((observation.x- landmark.x)*(observation.x- landmark.x)/
                                                (2*std_landmark[0]*std_landmark[0])) +
                               ((observation.y- landmark.y)*(observation.y- landmark.y)/
                                                (2*std_landmark[1]*std_landmark[1])));
            particles[i].weight *= gauss_norm*exp(-exponent);
        }
        //cout << particles[i].weight << endl;
        total_weight += particles[i].weight;
    }
    for (int i = 0; i < particles.size(); ++i) {
        particles[i].weight /= total_weight;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine generator;
    double max_weight = *max_element(weights.begin(), weights.end());
    vector<Particle> next_particles;
    double beta = 0;

    discrete_distribution<int> index_roulette(weights.begin(), weights.end());

    uniform_real_distribution<double> weight_d(0, 2*max_weight);
    for (int i = 0; i < num_particles; ++i) {
        int index = index_roulette(generator);
        beta += weight_d(generator);

        while (beta > weights[index]){
            beta -= weights[index];
            if(index < num_particles-1)
                index++;
            else
                index = 0;
        }
        next_particles.push_back(particles[index]);
    }    
    particles = next_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
     particle.sense_x = sense_x;
     particle.sense_y = sense_y;

     return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
