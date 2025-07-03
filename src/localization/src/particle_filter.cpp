#include <cmath>
#include <random>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"

struct Particle {
    double x, y, theta;
    double weight;
    
    Particle() : x(0), y(0), theta(0), weight(1.0) {}
};

class ParticleFilter : public rclcpp::Node {
public:
    ParticleFilter() : Node("particle_filter"), rng_(std::random_device{}()) {
        // Parameters
        num_particles_ = this->declare_parameter("num_particles", 500);
        wheel_radius_ = this->declare_parameter("wheel_radius", 0.033);
        wheelbase_ = this->declare_parameter("wheelbase", 0.287);
        
        // Minimal noise - just enough for particle diversity
        measurement_noise_v_ = this->declare_parameter("measurement_noise_v", 0.001);
        measurement_noise_omega_ = this->declare_parameter("measurement_noise_omega", 0.001);
        motion_noise_ = this->declare_parameter("motion_noise", 0.001);  // Very small
        
        initializeParticles();
        
        // Synchronized subscribers
        imu_sub_.subscribe(this, "/imu");
        joint_sub_.subscribe(this, "/joint_states");

        synchronizer_ = std::make_shared<message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Imu, sensor_msgs::msg::JointState>>>(
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Imu, sensor_msgs::msg::JointState>(10),
            imu_sub_, joint_sub_);

        synchronizer_->registerCallback(std::bind(
            &ParticleFilter::syncCallback, this, 
            std::placeholders::_1, std::placeholders::_2));
        
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf_state", 10);
        
        last_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "Particle filter initialized");
    }

private:
    std::vector<Particle> particles_;
    int num_particles_;
    double wheel_radius_, wheelbase_;
    double measurement_noise_v_, measurement_noise_omega_, motion_noise_;
    
    message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
    message_filters::Subscriber<sensor_msgs::msg::JointState> joint_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Imu, sensor_msgs::msg::JointState>>> synchronizer_;
    
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::mt19937 rng_;
    rclcpp::Time last_time_;

    void initializeParticles() {
        particles_.resize(num_particles_);
        
        // Small initial spread - particles start near origin
        std::uniform_real_distribution<double> pos_dist(-0.3, 0.3);
        std::uniform_real_distribution<double> angle_dist(-0.3, 0.3);
        
        for (auto& p : particles_) {
            p.x = pos_dist(rng_);
            p.y = pos_dist(rng_);
            p.theta = angle_dist(rng_);
            p.weight = 1.0 / num_particles_;
        }
    }

    void syncCallback(
        const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg,
        const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) {
        
        double dt = (this->now() - last_time_).seconds();
        if (dt <= 0.0 || dt > 0.5) {
            last_time_ = this->now();
            return;
        }
        
        // Extract control inputs from wheel encoders
        auto [v_linear, v_angular] = extractVelocities(joint_msg);
        double measured_omega = imu_msg->angular_velocity.z;
        
        //similar to EKF and KF

        predict(dt, v_linear, v_angular);
        update(v_linear, measured_omega);
        resample();
        publishState();
        
        last_time_ = this->now();
    }

    std::pair<double, double> extractVelocities(const sensor_msgs::msg::JointState::ConstSharedPtr joint_msg) {
        double v_left = 0.0, v_right = 0.0;
        
        for (size_t i = 0; i < joint_msg->name.size(); ++i) {
            if (joint_msg->name[i] == "wheel_left_joint") {
                v_left = joint_msg->velocity[i] * wheel_radius_;
            } else if (joint_msg->name[i] == "wheel_right_joint") {
                v_right = joint_msg->velocity[i] * wheel_radius_;
            }
        }
        
        // Differential drive kinematics
        double v_linear = (v_left + v_right) / 2.0;
        double v_angular = (v_right - v_left) / wheelbase_;
        
        return {v_linear, v_angular};
    }

    void predict(double dt, double v_linear, double v_angular) {
        std::normal_distribution<double> noise(0.0, motion_noise_);
        
        for (auto& p : particles_) {
            // Differential drive motion model
            if (std::abs(v_angular) < 1e-6) {
                // Straight line motion
                p.x += v_linear * cos(p.theta) * dt;
                p.y += v_linear * sin(p.theta) * dt;
            } else {
                // Curved motion
                double R = v_linear / v_angular;  // ICR
                double dtheta = v_angular * dt;
                
                p.x += R * (sin(p.theta + dtheta) - sin(p.theta));
                p.y += R * (cos(p.theta) - cos(p.theta + dtheta));
                p.theta += dtheta;
            }
            
            // Add minimal noise for particle diversity
            p.x += noise(rng_) * dt;
            p.y += noise(rng_) * dt;
            p.theta += noise(rng_) * dt;
            
            p.theta = normalizeAngle(p.theta);
        }
    }

    void update(double measured_v, double measured_omega) {
        double total_weight = 0.0;
        
        for (auto& p : particles_) {
            // For each particle, predict what the measurements should be
            // probability based on how well measurements match expectations
            double v_probability = exp(-0.5 * pow(measured_v - measured_v, 2) / pow(measurement_noise_v_, 2));
            double omega_probability = exp(-0.5 * pow(measured_omega - measured_omega, 2) / pow(measurement_noise_omega_, 2));
            p.weight = v_probability * omega_probability;
            total_weight += p.weight;
        }
        
        // Normalize weights
        if (total_weight > 0.0) {
            for (auto& p : particles_) {
                p.weight /= total_weight;
            }
        }
    }

    void resample() {
        // Systematic resampling
        std::vector<Particle> new_particles;
        new_particles.reserve(num_particles_);

        // Random starting point
        std::uniform_real_distribution<double> uniform(0.0, 1.0 / num_particles_);
        double u = uniform(rng_);

        // Initialize cumulative weight
        double cumulative_weight = particles_[0].weight;
        int j = 0;

        // Perform systematic resampling
        for (int i = 0; i < num_particles_; ++i) {
            double target = u + i * (1.0 / num_particles_);
            while (target > cumulative_weight && j < num_particles_ - 1) {
                j++;
                cumulative_weight += particles_[j].weight;
            }
            new_particles.push_back(particles_[j]);
            new_particles.back().weight = 1.0 / num_particles_; // Set equal weight
        }

        // Replace old particles with new resampled particles
        particles_ = std::move(new_particles);
    }

    void publishState() {
        // Compute weighted mean
        double mean_x = 0, mean_y = 0;
        double cos_sum = 0, sin_sum = 0;
        
        for (const auto& p : particles_) {
            mean_x += p.weight * p.x;
            mean_y += p.weight * p.y;
            cos_sum += p.weight * cos(p.theta);
            sin_sum += p.weight * sin(p.theta);
        }
        
        double mean_theta = atan2(sin_sum, cos_sum);
        
        auto odom = nav_msgs::msg::Odometry();
        odom.header.stamp = this->now();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        
        odom.pose.pose.position.x = mean_x;
        odom.pose.pose.position.y = mean_y;
        odom.pose.pose.orientation.z = sin(mean_theta / 2.0);
        odom.pose.pose.orientation.w = cos(mean_theta / 2.0);
        
        odom_pub_->publish(odom);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "PF: x=%.3f, y=%.3f, theta=%.3f", mean_x, mean_y, mean_theta);
    }

    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParticleFilter>());
    rclcpp::shutdown();
    return 0;
}