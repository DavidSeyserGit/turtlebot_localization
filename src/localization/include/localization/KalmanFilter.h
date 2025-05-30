#pragma once
#include "localization/FilterBase.h"

class KalmanFilter : public FilterBase
{
public:
    KalmanFilter();
    void predict(const Eigen::VectorXd &u, double dt) override;
    void update(const Eigen::VectorXd &z, const Eigen::MatrixXd &H) override;

private:
    Eigen::Vector3d x_;
    Eigen::Matrix3d P_;
    Eigen::Matrix3d Q_;
    Eigen::Matrix3d R_;
};