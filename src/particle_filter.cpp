/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
std::default_random_engine gen;

//#define _DEBUG_

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */


  normal_distribution <double> dist_x(x, std[0]);
  normal_distribution <double> dist_y(y, std[1]);
  normal_distribution <double> dist_t(theta, std[2]);

   
  num_particles = 100;  // TODO: Set the number of particles
  Particle p;
  for (int i = 0; i<num_particles; i++){
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_t(gen);
    p.weight = 1.0f;

    particles.push_back(p);

  }

  weights = vector<double>(num_particles, 1.0);
  is_initialized = true;

#ifdef _DEBUG_
  for (auto p: particles){
    //std::cout << "p.x: " << p.x << ", p.y: " << p.y << ", p.theta: " << p.theta<< std::endl;
  }
  std::cout << "inited" << std::endl;
  #endif
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  #ifdef _DEBUG_
  std::cout << "Prediction start" << std::endl;
  std::cout << "particles[0].x: " << particles[0].x << 
  ", particles[0].y: " << particles[0].y << 
  ", particles[0].theta: " << particles[0].theta<< std::endl;
  #endif 
  
    for (auto& p: particles){
      double x0 = p.x;
      double y0 = p.y;
      double t0 = p.theta;
    
    if (yaw_rate == 0.0f){
      x0 += velocity * delta_t * cos(p.theta);
      y0 += velocity * delta_t * sin(p.theta);
    }
    else{
      x0 += (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      y0 += (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      t0 += yaw_rate * delta_t;
    }

    normal_distribution <double> dist_x(x0, std_pos[0]);
    normal_distribution <double> dist_y(y0, std_pos[1]);
    normal_distribution <double> dist_t(t0, std_pos[2]);

    //adding noise
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_t(gen);
  }
  
#ifdef _DEBUG_
  std::cout << "Prediction start" << std::endl;
  std::cout << "particles[0].x: " << particles[0].x << 
  ", particles[0].y: " << particles[0].y << 
  ", particles[0].theta: " << particles[0].theta<< std::endl;
  std::cout << "Prediction end" << std::endl;
  #endif 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  #ifdef _DEBUG_
  std::cout << "dataAssociation start" << std::endl;
  #endif 

  for (auto& obs_i : observations){
    double min_Dist = std::numeric_limits<double>::max();
    int mapNo = -1;
    for (auto& pred_i : predicted){
      double xdist = obs_i.x - pred_i.x;
      double ydist = obs_i.y - pred_i.y;
      double curr_Dist = sqrt (xdist*xdist + ydist*ydist);

      if (curr_Dist < min_Dist){
        mapNo = pred_i.id;
        min_Dist = curr_Dist;
      }

    }
    obs_i.id = mapNo;
  }
  
#ifdef _DEBUG_
  std::cout << "dataAssociation end" << std::endl;
  #endif 
}

double multivariateGaussian(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y){
  double dx = x-mu_x;
  double dy = y-mu_y;
  return ( 1/(2*M_PI*sig_x*sig_y)) * exp( -( dx*dx/(2*sig_x*sig_x) + (dy*dy/(2*sig_y*sig_y))));
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  #ifdef _DEBUG_
  std::cout << "updateWeights start" << std::endl;
  #endif 
  
  int ind=-1;  
  for(auto& p: particles){
    ind++;

    vector<LandmarkObs> in_range;
    for (auto& maplm: map_landmarks.landmark_list){
      double dx = p.x - maplm.x_f;
      double dy = p.y - maplm.y_f;
      if ( dx*dx + dy*dy <= (sensor_range*sensor_range) ) {
        in_range.push_back(LandmarkObs{ maplm.id_i, maplm.x_f, maplm.y_f });
      }
    }

    // coordinate transformation 
    vector<LandmarkObs> obs_on_map;
    for (auto& obs:observations){
      double x = cos(p.theta)*obs.x - sin(p.theta)*obs.y + p.x;
      double y = sin(p.theta)*obs.x + cos(p.theta)*obs.y + p.y;
      obs_on_map.push_back(LandmarkObs{ obs.id, x, y});
    }

    // associating observation to landmark.
    dataAssociation(in_range, obs_on_map);

    
    // calculating new weight
    p.weight = 1.0;
    for(auto& obs : obs_on_map){

      double lm_x, lm_y;
      for (auto& lm : in_range){
        if ( lm.id == obs.id) {
          lm_x = lm.x;
          lm_y = lm.y;
          break;
        }
      }

      double weight = multivariateGaussian(obs.x, obs.y, lm_x, lm_y, std_landmark[0], std_landmark[1]);
      if (weight == 0) {
        p.weight *= 0.00001;
      } else {
        p.weight *= weight;
      }
      weights[ind] = p.weight;
    }
  }
  #ifdef _DEBUG_
  std::cout << "Updateweights end" << std::endl;
  #endif 
}


void ParticleFilter::resample() {

  vector<Particle> newParticles;

  std::uniform_real_distribution<double> dist_0_1(0.0, 1.0);

  // resampling with replacement, propotion to the weight of the particles
  int ind = rand() % num_particles;
  double weight_max = *std::max_element(weights.begin(), weights.end());
  double beta = 0.0;

  for (int i = 0; i < num_particles; ++i) {
    beta += dist_0_1(gen) * 2.0 * weight_max;
    while(beta > weights[ind]) {
      beta -= weights[ind];
      ind = (ind + 1) % num_particles;
    }
    newParticles.push_back(particles[ind]);
  }

  particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


