/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * updated by Srinivas S
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

#include <map>    // for resample
#include <random> // for resample

#include "helper_functions.h"
//#include "map.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to first position
   * (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method (and others in this file).

   * initial position is given and all particles should be initialized by sampling from a gaussian
   * distribution centered around that particle
   */
  num_particles = 100; //change later // TODO: Set the number of particles


  using std::normal_distribution;
  std::default_random_engine gen;

  normal_distribution<double> dist_x(0, std[0]);  //x std_x = std[0]; // Set std deviations for x, y,theta
  normal_distribution<double> dist_y(0, std[1]); //y
  normal_distribution<double> dist_theta(0, std[2]); //theta

  for (int i = 0; i < num_particles; ++i) {
    Particle init_particle;

    init_particle.id = i;
    init_particle.x = x + dist_x(gen); //x + std[0];
    init_particle.y = y + dist_y(gen); //y + std[1];
    init_particle.theta = theta + dist_theta(gen); //theta + std[2];
    init_particle.weight = 1.0;

    particles.push_back(init_particle);
    weights.push_back(init_particle.weight); //?? should this vector be initialized here?
  }
  is_initialized = true;
  //particles[i].weight > highest_weight)
  cout << "done initializing." << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   *
   * using these measurements, update each particles position estimate, account for sensor noise
   * add gaussian noise by sampling from gau dist mean = updated particle position, std dev = std dev of measurements
   */
  //cout << "begin prediction.." << endl;
  //cout << "delta_t: " << delta_t <<"velocity: "<< velocity<< " yaw_rate: "<< yaw_rate <<endl;
  using std::normal_distribution;
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; ++i) {
     double x_n, y_n, theta_n;

    if(fabs(yaw_rate) < 0.00001) { //need to add this to cover for yaw_rate=0 case
      x_n = particles[i].x + (velocity*delta_t)*(cos(particles[i].theta));
      y_n = particles[i].y + (velocity*delta_t)*(sin(particles[i].theta));
      theta_n = particles[i].theta;
      //cout<<"yaw_rate == 0 x: "<<particles[i].x<<" y: "<<particles[i].y<<" theta: "<<particles[i].theta<<endl;
    } else { //if(yaw_rate != 0) {
      x_n = particles[i].x + (velocity/yaw_rate)*( sin(particles[i].theta+ yaw_rate*delta_t)-sin(particles[i].theta) );
      y_n = particles[i].y + (velocity/yaw_rate)*( cos(particles[i].theta)-cos(particles[i].theta+ yaw_rate*delta_t) );
      theta_n = particles[i].theta + (yaw_rate*delta_t);
      //cout<< "yaw_rate != 0 x: "<<particles[i].x<<" y: "<<particles[i].y<<" theta: "<<particles[i].theta<<endl;
    }
    normal_distribution<double> dist_x(0, std_pos[0]);       //x_n  //std_x = std_pos[0]; // Set std deviations for x, y,theta
    normal_distribution<double> dist_y(0, std_pos[1]);      //y_n   //std_y
    normal_distribution<double> dist_theta(0, std_pos[2]);  //theta_n
    //either add noise to x,y,theta or sample from a dist with mean to x, y, theta
    //randomly update x,y, theta for each particle to simulate real time behavior of the car location
    //based on landmark observations. average of all these observations is the car position
    //this is updated based on the update step
    particles[i].x = x_n + dist_x(gen);
    particles[i].y = y_n + dist_y(gen);
    particles[i].theta = theta_n + dist_theta(gen);
    //std::cout<<"x: "<<particles[i].x<<" y: "<<particles[i].y<<" theta: "<<particles[i].theta<<std::endl;
  }
  cout << "end prediction.." << endl;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each observed measurement and assign the observed
   * measurement to this particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
   * implement this method and use it as a helper during the updateWeights phase.
   */
  // this func performs nearest neighbor data association and assign each sensor observation, map landmark id associated with it
  // predicted has the map landmarks closet to the particle
  // observations is the transformed observation vector - in this the obs are transformed into particle map coordinates
  // associate each transformed obs with a map landmark by setting the map id in the observations vector
  for(int i=0; i < observations.size(); i++) {
    int curMapIndex = -1;
    double minDist = 100000;
    for(int j=0; j < predicted.size(); j++) {

      double MapTransObsDist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(MapTransObsDist <= minDist) { //find the map landmark location closest to the transformed observation of particle
        minDist = MapTransObsDist;
        curMapIndex = predicted[j].id;
      }
    }
    observations[i].id = curMapIndex; // every observation will be associated with a map landmark
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations, const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian
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
   *
   * take sensor range, landmark obs uncertainities, landmark obs and map landmark
   * predict meas to all map landmarks within sensor range for each particle
   * once have predicted landmark meas, can use the data asso func to associate sensor meas to map landmarks
   * need this association to calc new weight of each particle by using multi variate gau pdf
   * finally normalize weights so that they are between 0-1, use weights as prob for resampling
    */
  // sensor_range = 50;  sigma_landmark [2] = {0.3, 0.3} == std_landmark
  //vector<LandmarkObs> noisy_observations; has x,y values of all particles
  //observations = obs values of landmarks in vehicle cos
  //map_landmarks = actual map landmarks

  //1) calc dist from particle to each map landmark, if this is less than sensor range, store that map landmark
  //2) transform observations to map coordinates in particle reference frame
    //3) remove all obs out of sensor range -- optional
  //4) dataAssociation
  //5) for all observations, find closest map landmark to a transformed observation value and update the
  // id value in observations struct (its missing from the simulator call for this reason) -- dataAssociation
  //6) calculate new weight
  //7) normalize the weights -- optional?

  cout << "begin update weights.." << endl;
  for (int i = 0; i < num_particles; i++) {

    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    //1) calc dist from particle to each map landmark, if this is less than sensor range, store that map landmark
    vector<LandmarkObs> predicted_landmarks;
    for(int j=0; j< map_landmarks.landmark_list.size(); j++) {

      LandmarkObs temp_map_pos;
      double map_id = map_landmarks.landmark_list[j].id_i;
      double map_x = map_landmarks.landmark_list[j].x_f;
      double map_y = map_landmarks.landmark_list[j].y_f;
      //circular or ractangular range does not make a difference 
      //if( (fabs(map_x - p_x) <= sensor_range) && (fabs(map_y - p_y) <= sensor_range) ) {
      double tempDist = dist(particles[i].x, particles[i].y, map_x, map_y); //
      if(tempDist <= sensor_range) {
        //this map landmark is within sensor range of the particle
        temp_map_pos.id = map_id;
        temp_map_pos.x = map_x;
        temp_map_pos.y = map_y;
        predicted_landmarks.push_back(temp_map_pos);//map_landmarks.landmark_list[j]);
      }
    } //for(int j=0; j<map_landmarks.landmark_list.size(); j++) {

    // 2) transform observations to map coordinate in particle reference frame
    vector<LandmarkObs> transformed_observations;
    for(int k=0; k < observations.size(); k++) {

      double x_obs, y_obs; // initial observations
      double x_TObs, y_TObs; // transformed observations
      LandmarkObs tempObs;
      //p_theta = particles[i].theta;
      x_obs = observations[k].x + std_landmark[0]; //add gaussian noise to observation
      y_obs = observations[k].y + std_landmark[1];

      x_TObs = p_x + (cos(p_theta)*x_obs)-(sin(p_theta)*y_obs); // transform to map x coordinate
      y_TObs = p_y + (sin(p_theta)* x_obs)+(cos(p_theta)*y_obs); // transform to map y coordinate
      ///////////////experimental - confirm this
      //eliminate all observations out of sensor range for this particle
      //otherwise every observation will be mapped to a landmark in some way, may not need all observations
      //3) remove all obs out of sensor range -- optional
      //does not seem to affect the result a lot even if this check is not performed, though it seems that
      // all observations outside sensor range are irrelevant
      double tempDist = dist(p_x, p_y, x_TObs, y_TObs);
      if(tempDist <= sensor_range) {
        //this observation is within sensor range of the particle
        tempObs.x = x_TObs;
        tempObs.y = y_TObs;
        transformed_observations.push_back(tempObs);
      }
    } //for(int j=0; j < observations.size(); j++) {
    //4)
    dataAssociation(predicted_landmarks, transformed_observations);

    particles[i].weight = 1.0; //This is a mandatory step without which the error increases a lot
    vector<int> associations_1; // to set association
    vector<double> sense_x_1;
    vector<double> sense_y_1;

    //5) for all observations, find closest map landmark to a transformed observation value and update the
    // id value in observations struct (its missing from the simulator call for this reason) -- dataAssociation
    for(int m=0; m < transformed_observations.size(); m++) {
      int tobs_id = transformed_observations[m].id;
      double x_obs = transformed_observations[m].x;
      double y_obs = transformed_observations[m].y;

      double mu_x, mu_y; //mu_x, mu_y is position on  map
      for(int n=0; n < predicted_landmarks.size(); n++) {
         int map_id = predicted_landmarks[n].id;
         if(tobs_id == map_id) {
           mu_x = predicted_landmarks[n].x;
           mu_y = predicted_landmarks[n].y;
         } //if(tobs_id == map_id) {
       } //for(int n=0; n < predicted_landmarks.size(); n++) {

      //6) calculate the weight
      //from data association, this map landmark was closest to this transformed observation
      double obsWeight = multiv_prob(std_landmark[0], std_landmark[1], x_obs, y_obs, mu_x, mu_y);
      //multiply the weight of each observation to get final weight for the particle
      if(obsWeight > 0) {
        particles[i].weight *= obsWeight;
      } else {
        particles[i].weight *= 0.00001;//add negligible weight
      }
      //now we know to which map landmark this observation is closest to
      //the particle will be associated with map landmarks id and transformed observation x,y location
      associations_1.push_back(tobs_id); //the id that matches the map id
      sense_x_1.push_back(x_obs);
      sense_y_1.push_back(y_obs);

    } //for(int m=0; m < transformed_observations.size(); m++) {
    //associate particle_predictions with a map landmark
    SetAssociations(particles[i], associations_1, sense_x_1, sense_y_1);

  } //for (int i = 0; i < num_particles; ++i) {

  //7) normalize the weights -- optional?
  double total_weight = 0;
  for (int i = 0; i < num_particles; ++i) {
    total_weight += particles[i].weight;
  }
  cout << "total_weight: " << total_weight << endl;
  for (int i = 0; i < num_particles; ++i) {
    //std::cout << "particles[" << i << "].weight before: " << particles[i].weight << std::endl;
    particles[i].weight = particles[i].weight/total_weight;
    //std::cout << "particles[" << i << "].weight after: " << particles[i].weight << std::endl;
  }
  cout << "end update weights.." << endl;
 }

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   * Use the weight of particle in particle filter and the c++ std lib func to update particles to the bayesian posterior distribution
   * make vector/array of all 50 particle weights
   * run the loop num_particles times
   * randomly one weight (index) is picked, the probability of picking the weight is equal to its magnitude (discrete_distribution)
   * create a new particle array/vector and add that particle at that weight index to this
   * pass weights from particle vector to an array and then to
   * after resampling, the particles vector has particles with only chosen weights
   */
  weights.clear();
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  }
  std::default_random_engine gen;
  std::discrete_distribution<> distribution(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles = {};
  for (int i = 0; i < num_particles; ++i) {
    int selected_idx = distribution(gen);
    resampled_particles.push_back(particles[selected_idx]);
  }
  particles = resampled_particles;

} //end resample

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
 /**
  * particle: the particle to which assign each listed association,
  *  and association's (x,y) world coordinates mapping
  * associations: The landmark id that goes along with each listed association
  * sense_x: the associations x mapping already converted to world coordinates
  * sense_y: the associations y mapping already converted to world coordinates
  */
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
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

  //cout << "..getSenseCoord.." << s << endl;
  return s;
}

