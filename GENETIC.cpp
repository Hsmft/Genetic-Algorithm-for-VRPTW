#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <queue>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <atomic>

using namespace std;

// Global variable for controlling algorithm termination
atomic<bool> stopAlgorithm(false);
bool timeLimitMessagePrinted = false;

// Customer structure
struct Customer {
    int id;
    double x, y;
    int demand;
    double serviceTime;
    double earliest;
    double latest;
};

// Route structure
struct Route {
    vector<int> customerIds;
    double totalDistance;
    vector<double> arrivalTimes;
    vector<double> departureTimes;
    int load;
    vector<double> min_slacks; // Minimum slack time from each position to the end
    Route() : totalDistance(0.0), load(0) {}
};

// Problem data structure
struct ProblemData {
    vector<Customer> customers;
    int vehicleCapacity;
    int maxVehicles;
    double maxRouteDuration;
    vector<vector<double>> distanceMatrix;
    unordered_map<int, int> idToIndex;
};

// Individual structure for population
struct Individual {
    vector<int> sequence;};

// Helper function to print solution information
void printSolutionInfo(const vector<Route>& routes, const ProblemData& data, const string& label) {
    int vehicles = 0;
    double totalDistance = 0.0;
    for (const auto& route : routes) {
        if (!route.customerIds.empty()) {
            vehicles++;
            totalDistance += route.totalDistance;}}
    cout << label << ": Vehicles = " << vehicles << ", Total Distance = " << fixed << setprecision(2) << totalDistance << endl;}

// Check time limit
bool checkTimeLimit(chrono::steady_clock::time_point startTime, int maxTime) {
    if (stopAlgorithm) return false;
    auto currentTime = chrono::steady_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::seconds>(currentTime - startTime).count();
    if (maxTime > 0 && elapsedTime >= maxTime) {
        stopAlgorithm = true;
        if (!timeLimitMessagePrinted) {
            cout << "Time limit reached: " << elapsedTime << " seconds" << endl;
            timeLimitMessagePrinted = true;}
    return false;}
    return true;}

// Read instance file and calculate distance matrix
ProblemData readInstance(const string& filename) {
    ProblemData data;
    ifstream infile(filename);
    if (!infile) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);}
    string line;
    while (getline(infile, line)) {
        if (line.find("CUST NO.") != string::npos) {
            getline(infile, line);
            break;
        } else if (line.find("NUMBER") != string::npos) {
            getline(infile, line);
            istringstream iss(line);
            iss >> data.maxVehicles >> data.vehicleCapacity;
        }}
    while (getline(infile, line)) {
        istringstream issCust(line);
        Customer cust;
        if (issCust >> cust.id >> cust.x >> cust.y >> cust.demand  >> cust.earliest >>  cust.latest >>cust.serviceTime) {
            data.customers.push_back(cust);
        } else {
            break;
        }}
    data.maxRouteDuration = data.customers[0].latest;
    for (size_t i = 0; i < data.customers.size(); ++i) {
        data.idToIndex[data.customers[i].id] = i;}
    size_t n = data.customers.size();
    data.distanceMatrix.resize(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                data.distanceMatrix[i][j] = 0.0;
            } else {
                double dx = data.customers[i].x - data.customers[j].x;
                double dy = data.customers[i].y - data.customers[j].y;
                data.distanceMatrix[i][j] = sqrt(dx * dx + dy * dy);}}}
    infile.close();
    return data;}

// Check if a customer can be inserted into a route at a specified position
bool canInsert(const Route& route, int pos, int customerId, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return false;
    // Convert customer ID to index in the customers list for quick access
    int index = data.idToIndex.at(customerId);
    // Determine the index of the previous customer (depot if position is 0)
    int prevIndex = (pos == 0) ? 0 : data.idToIndex.at(route.customerIds[pos - 1]);
    // Determine the index of the next customer (depot if position is at the end)
    int nextIndex = (pos == route.customerIds.size()) ? 0 : data.idToIndex.at(route.customerIds[pos]);
    int newLoad = route.load + data.customers[index].demand;
    if (newLoad > data.vehicleCapacity) return false;
    //Calculate the time increase due to inserting the customer (travel and service time)
    double lowerDelta = data.distanceMatrix[prevIndex][index] + 
    data.customers[index].serviceTime + data.distanceMatrix[index][nextIndex] - data.distanceMatrix[prevIndex][nextIndex];
   //Check slack time constraint; return false if the time increase exceeds minimum slack
    if (pos < route.min_slacks.size() && lowerDelta > route.min_slacks[pos]) {
        return false;}
    // Calculate arrival time to the new customer (from depot if first customer)
    double arrivalTime = (pos == 0) ? data.distanceMatrix[0][index] : route.departureTimes[pos - 1] + data.distanceMatrix[prevIndex][index];
    double serviceStartTime = max(arrivalTime, data.customers[index].earliest);
    if (serviceStartTime > data.customers[index].latest) return false;
    double departureTime = serviceStartTime + data.customers[index].serviceTime;
    double currentTime = departureTime;
    // Evaluate the impact of insertion on subsequent customers in the route
    for (size_t i = pos; i < route.customerIds.size(); ++i) {
        int nextCustIndex = data.idToIndex.at(route.customerIds[i]);
        double travelTime = data.distanceMatrix[index][nextCustIndex];
        arrivalTime = currentTime + travelTime;
        serviceStartTime = max(arrivalTime, data.customers[nextCustIndex].earliest);
        if (serviceStartTime > data.customers[nextCustIndex].latest) return false;
        //update current time with the service time of the next customer
        currentTime = serviceStartTime + data.customers[nextCustIndex].serviceTime;
        index = nextCustIndex;}
    double returnTime = currentTime + data.distanceMatrix[index][0];
    if (returnTime > data.maxRouteDuration) return false;
    return true;}

// Check route feasibility
bool isRouteFeasible(const Route& route, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return false;
    int capacityUsed = 0;
    double currentTime = 0.0;
    int currentIndex = 0;
    for (int custId : route.customerIds) {
        if (!isFinal && !checkTimeLimit(startTime, maxTime)) return false;
        int index = data.idToIndex.at(custId);
        capacityUsed += data.customers[index].demand;
        if (capacityUsed > data.vehicleCapacity) return false;
        double travelTime = data.distanceMatrix[currentIndex][index];
        double arrivalTime = currentTime + travelTime;
        double serviceStartTime = max(arrivalTime, data.customers[index].earliest);
        if (serviceStartTime > data.customers[index].latest) return false;
        currentTime = serviceStartTime + data.customers[index].serviceTime;
        currentIndex = index;}
    double returnTravelTime = data.distanceMatrix[currentIndex][0];
    currentTime += returnTravelTime;
    if (currentTime > data.maxRouteDuration) return false;
    return true;}

// Check solution feasibility
bool isSolutionFeasible(const vector<Route>& routes, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return false;
    int usedVehicles = 0;
    for (const auto& r : routes) {
        if (!r.customerIds.empty()) 
            ++usedVehicles;}
    if (usedVehicles > data.maxVehicles)
        return false;
    unordered_set<int> visitedCustomers;
    for (const auto& route : routes) {
        if (!isRouteFeasible(route, data, startTime, maxTime, isFinal))
            return false;
        for (int custId : route.customerIds) {
            if (visitedCustomers.count(custId))
                return false;
            visitedCustomers.insert(custId);}}
    for (size_t i = 1; i < data.customers.size(); ++i) {
        if (!visitedCustomers.count(data.customers[i].id))
            return false;}
    return true;}

// Update route
void updateRoute(Route& route, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return;
    route.arrivalTimes.clear();
    route.departureTimes.clear();
    route.load = 0;
    route.totalDistance = 0;
    route.min_slacks.clear();
    if (route.customerIds.empty()) return;
    double currentTime = 0.0;
    route.arrivalTimes.push_back(currentTime);
    int currentIndex = 0;
    for (int custId : route.customerIds) {
        if (!isFinal && !checkTimeLimit(startTime, maxTime)) return;
        int index = data.idToIndex.at(custId);
        double travelTime = data.distanceMatrix[currentIndex][index];
        currentTime += travelTime;
        route.totalDistance += travelTime;
        if (currentTime < data.customers[index].earliest) currentTime = data.customers[index].earliest;
        route.arrivalTimes.push_back(currentTime);
        currentTime += data.customers[index].serviceTime;
        route.departureTimes.push_back(currentTime);
        route.load += data.customers[index].demand;
        currentIndex = index;}
    double returnDist = data.distanceMatrix[currentIndex][0];
    route.totalDistance += returnDist;
    currentTime += returnDist;
    route.arrivalTimes.push_back(currentTime);
    route.departureTimes.push_back(currentTime);
    route.min_slacks.resize(route.customerIds.size() + 1);
    route.min_slacks.back() = numeric_limits<double>::max();
    for (int i = route.customerIds.size() - 1; i >= 0; --i) {
        int index = data.idToIndex.at(route.customerIds[i]);
        double slack = data.customers[index].latest - route.arrivalTimes[i];
        route.min_slacks[i] = min(slack, route.min_slacks[i + 1]);}}

// Objective function
pair<int, double> objectiveFunction(const vector<Route>& routes, long long& evalCount, int maxEvaluations, 
    chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return {0, 0.0};
    if (evalCount >= maxEvaluations) {
        stopAlgorithm = true;
        return {0, 0.0};}
    evalCount++;
    int vehicles = 0;
    double totalDistance = 0.0;
    for (const auto& route : routes) {
     if (!isFinal && !checkTimeLimit(startTime, maxTime)) return {0, 0.0};
        if (!route.customerIds.empty()) {
            vehicles++;
            totalDistance += route.totalDistance;}}
    return {vehicles, totalDistance};}

//Calculate the cost of inserting a customer into a route at a given position
double calculateInsertionCost(const Route& route, int pos, int customer, const ProblemData& data) {
    // Set previous customer (depot if pos = 0)
    int prev = (pos == 0) ? 0 : route.customerIds[pos - 1];
    // Set next customer (depot if pos is at the end)
    int next = (pos == route.customerIds.size()) ? 0 : route.customerIds[pos];
    // Map IDs to indices for quick access
    int index_prev = data.idToIndex.at(prev);
    int index_next = data.idToIndex.at(next);
    int index_cust = data.idToIndex.at(customer);
    // Calculate spatial cost (change in distance due to insertion)
    double spatialCost = data.distanceMatrix[index_prev][index_cust] + data.distanceMatrix[index_cust][index_next] - 
     data.distanceMatrix[index_prev][index_next];
    // Initialize temporal cost
    double temporalCost = 0.0;
    // Calculate temporal cost if not inserting at the start
    if (pos > 0) {
        double arrivalTime = route.departureTimes[pos - 1] + data.distanceMatrix[index_prev][index_cust];
        temporalCost = max(0.0, data.customers[index_cust].earliest - arrivalTime);}
    // Weight factor for combining costs
    double alpha = 0.5;
    // Return combined cost (spatial and temporal)
    return alpha * spatialCost + (1 - alpha) * temporalCost;}

// Reconstruct routes
vector<Route> reconstructRoutes(
    const vector<int>& sequence, 
    const ProblemData& data, 
    chrono::steady_clock::time_point startTime, 
    int maxTime, 
    bool isFinal = false) {
    // if time exceeded and not final pass, abort
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return {};
    vector<Route> routes;
    // iterate through each customer in the sequence
    for (int customer : sequence) {
    if (!isFinal && !checkTimeLimit(startTime,maxTime)) return routes;
        // track best insertion location
        double bestCost = numeric_limits<double>::max();
        int bestRouteIndex = -1;
        int bestPosition = -1;
        int idxCust = data.idToIndex.at(customer);
        // try to insert into existing routes
      for (size_t r = 0; r < routes.size(); ++r) {
            if (routes[r].customerIds.empty()) continue;
            // find the closest customer in route by spatial+temporal metric
            int closestIndex = -1;
            double minCombinedDist = numeric_limits<double>::max();
            for (size_t i = 0; i < routes[r].customerIds.size(); ++i) {
                int idxI = data.idToIndex.at(routes[r].customerIds[i]);
                double spatialDist = data.distanceMatrix[idxCust][idxI];
                double temporalDist = abs(
                    data.customers[idxCust].earliest - 
                    data.customers[idxI].earliest);
                double combinedDist = 0.5 * spatialDist + 0.5 * temporalDist;
                if (combinedDist < minCombinedDist) {
                    minCombinedDist = combinedDist;
                    closestIndex = i;}}
            // check insertion at closestIndex and the next position
            vector<int> positionsToCheck = { closestIndex, closestIndex + 1 };
            for (int pos : positionsToCheck) {
                if (pos < 0 || pos > (int)routes[r].customerIds.size()) continue;
                //test time, capacity & time-window feasibility
                if (canInsert(routes[r], pos, customer, data, startTime, maxTime, isFinal)) {
                    double cost = calculateInsertionCost(routes[r], pos, customer, data);
                    // update best if cost improves
                 if (cost < bestCost) {
                        bestCost = cost;
                        bestRouteIndex = r;
                        bestPosition = pos;}}}}
        if (bestRouteIndex != -1) {
            // insert into the best existing route
            auto& bestRoute = routes[bestRouteIndex];
            bestRoute.customerIds.insert(
                bestRoute.customerIds.begin() + bestPosition, 
                customer);
            updateRoute(bestRoute, data, startTime, maxTime, isFinal);
} else {
            // create new route if no feasible insertion found
            Route newRoute;
            newRoute.customerIds.push_back(customer);
            if (isRouteFeasible(newRoute, data, startTime, maxTime, isFinal)) {
                routes.push_back(newRoute);
                updateRoute(routes.back(), data, startTime, maxTime, isFinal);
            } else {
                cout << "Warning: Customer " << customer 
                     << " cannot be assigned to any route." << endl;}}}
    return routes;}

//Repair an invalid solution by reinserting unassigned customers
vector<Route> repairSolution(
    const vector<Route>& routes, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    //copy existing routes to modify
    vector<Route> repairedRoutes = routes;
    //collect all customers except depot
    unordered_set<int> unassignedCustomers;
    for (size_t i = 1; i < data.customers.size(); ++i) {
        unassignedCustomers.insert(data.customers[i].id);}
    // remove those already in routes
    for (auto& route : repairedRoutes) {
        for (int cid : route.customerIds) {
            unassignedCustomers.erase(cid);}}
    // reinsert until none left
    while (!unassignedCustomers.empty()) {
        int customer = *unassignedCustomers.begin();
        unassignedCustomers.erase(customer);
        double bestCost = numeric_limits<double>::max();
        int bestRouteIndex = -1;
        int bestPosition   = -1;
        int idxCust        = data.idToIndex.at(customer);
        // try inserting into each existing route
        for (size_t r = 0; r < repairedRoutes.size(); ++r) {
            if (repairedRoutes[r].customerIds.empty()) continue;
            // find closest customer by spatial+temporal metric
            int closestIdx = -1;
            double minCombined = numeric_limits<double>::max();
            for (size_t i = 0; i < repairedRoutes[r].customerIds.size(); ++i) {
                int idxI = data.idToIndex.at(repairedRoutes[r].customerIds[i]);
                double spatial = data.distanceMatrix[idxCust][idxI];
                double temporal = abs(
                    data.customers[idxCust].earliest -
                    data.customers[idxI].earliest);
                double combined = 0.5 * spatial + 0.5 * temporal;
                if (combined < minCombined) {
                    minCombined = combined;
                    closestIdx  = i;}}
            // check insertion at closestIdx and next position
            vector<int> positionsToCheck = { closestIdx, closestIdx + 1 };
            for (int pos : positionsToCheck) {
                if (pos < 0 || pos > (int)repairedRoutes[r].customerIds.size()) continue;
                // ensure capacity, time windows, and time limit
                if (canInsert(repairedRoutes[r], pos, customer, data, startTime, maxTime, isFinal)) {
                    double cost = calculateInsertionCost(
                        repairedRoutes[r], pos, customer, data);
                    // update best if improved
                    if (cost < bestCost) {
                        bestCost       = cost;
                        bestRouteIndex = r;
                        bestPosition   = pos;}}}}
        if (bestRouteIndex != -1) {
            // insert into best found route
            auto& route = repairedRoutes[bestRouteIndex];
            route.customerIds.insert(
                route.customerIds.begin() + bestPosition,
                customer);
            updateRoute(route, data, startTime, maxTime, isFinal);
        } else {
            // no feasible insertion: start a new route
            Route newRoute;
            newRoute.customerIds.push_back(customer);
            if (isRouteFeasible(newRoute, data, startTime, maxTime, isFinal)) {
                repairedRoutes.push_back(newRoute);
                updateRoute(repairedRoutes.back(), data, startTime, maxTime, isFinal);
            } else {
                // still infeasible after repair
                cout << "Warning: Customer " << customer 
                     << " cannot be assigned even after repair." << endl;}}}
    return repairedRoutes;}


// Evaluate solution
double evaluateSolution(const Individual& ind, const ProblemData& data, long long& evalCount, 
    int maxEvaluations, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return numeric_limits<double>::max();
    if (stopAlgorithm) return numeric_limits<double>::max();
    vector<Route> routes = reconstructRoutes(ind.sequence, data, startTime, maxTime, isFinal);
    if (!isSolutionFeasible(routes, data, startTime, maxTime, isFinal)) {routes = repairSolution(routes, data, startTime, maxTime, isFinal);
        if (!isSolutionFeasible(routes, data, startTime, maxTime, isFinal)) {
            return numeric_limits<double>::max();}}
    auto [vehicles, distance] = objectiveFunction(routes, evalCount, maxEvaluations, startTime, maxTime, isFinal);
    if (stopAlgorithm) return numeric_limits<double>::max();
    return vehicles * 100.0 + 0.001 * distance;}

// Functions for selecting initial customer
int findFarthestCustomer(const ProblemData& data, const vector<int>& unrouted) {
    double maxDist = 0;
    int selected = -1;
    for (int id : unrouted) {int index = data.idToIndex.at(id);
      double dist = data.distanceMatrix[0][index];
        if (dist > maxDist) {
            maxDist = dist;
            selected = id;}}
    return selected;}
int findEarliestCustomer(const ProblemData& data, const vector<int>& unrouted) {double minEarliest = numeric_limits<double>::max();
    int selected = -1;
    for (int id : unrouted) {
     int index = data.idToIndex.at(id);
        double earliest = data.customers[index].earliest;
        if (earliest < minEarliest) {
            minEarliest = earliest;
            selected = id;}}
    return selected;}
int findLatestCustomer(const ProblemData& data, const vector<int>& unrouted) {double minLatest = numeric_limits<double>::max();
    int selected =-1;
    for (int id : unrouted)   { int index = data.idToIndex.at(id);
        double latest = data.customers[index].latest;
        if (latest < minLatest) {
            minLatest = latest;
            selected = id;}}
    return selected;}
int findRandomCustomer(const ProblemData& data, const vector<int>& unrouted, mt19937& rng) {
    uniform_int_distribution<size_t> dist(0, unrouted.size() - 1);
    return unrouted[dist(rng)];   }

//Greedy construction of sequence
vector<int> greedyConstruction(const ProblemData& data, const string& criterion, mt19937& rng, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return {};
    unordered_set<int> unrouted;
    for (size_t i = 1; i < data.customers.size(); ++i) {
        unrouted.insert(data.customers[i].id);}
    vector<Route> routes;
    while (!unrouted.empty()) {
    int seed;
        if (criterion == "farthest") {
            seed = findFarthestCustomer(data, vector<int>(unrouted.begin(),unrouted.end()));
        } else if (criterion == "earliest") {
            seed = findEarliestCustomer(data, vector<int>(unrouted.begin(), unrouted.end()));}
             else if (criterion == "latest") {
            seed = findLatestCustomer(data, vector<int>(unrouted.begin(),unrouted.end()));}
             else {
            seed = findRandomCustomer(data, vector<int>(unrouted.begin(), unrouted.end()), rng);}
        if (seed == -1) break ;
        Route currentRoute;
        currentRoute.customerIds.push_back(seed);
        updateRoute(currentRoute, data, startTime, maxTime, isFinal);
        unrouted.erase(seed);
        while (true) {
            double bestCost = numeric_limits<double>::max();
            int bestCustomer = -1 ;
            int bestPosition =-1;
            for (int customer : unrouted) {
                for (size_t pos = 0; pos <= currentRoute.customerIds.size(); ++pos) {
                    if (canInsert(currentRoute, pos, customer, data, startTime, maxTime, isFinal)) {
                        double cost = calculateInsertionCost(currentRoute, pos, customer, data);
                        if (cost < bestCost) {bestCost = cost;bestCustomer = customer;bestPosition = pos;}}}}
            if (bestCustomer != -1) {
                currentRoute.customerIds.insert(currentRoute.customerIds.begin() + bestPosition, bestCustomer);
                updateRoute(currentRoute, data, startTime, maxTime, isFinal);
                unrouted.erase(bestCustomer);
            } else {
                break;}}
        routes.push_back(currentRoute);}
    vector<int> sequence;
    for (const auto& route : routes) {
        sequence.insert(sequence.end(), route.customerIds.begin(), route.customerIds.end());}
    return sequence;}

// Generate initial population
vector<Individual> generateInitialPopulation(const ProblemData& data, mt19937& rng, 
    int populationSize, chrono::steady_clock::time_point startTime, int maxTime, vector<double>& fitness, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return {};
    
    vector<Individual> population;
    vector<int> customers;
    for (size_t i = 1; i < data.customers.size(); ++i) {
        customers.push_back(data.customers[i].id);}
    vector<string> criteria = {"farthest", "earliest", "latest", "random"};
    int numGreedy = populationSize / 3;
    int numRandom = populationSize - numGreedy;
    for (int i = 0; i < numGreedy; ++i) {
        if (!isFinal && !checkTimeLimit(startTime, maxTime)) return population;
        Individual ind;
        string criterion = criteria[i % criteria.size()];
        ind.sequence = greedyConstruction(data, criterion, rng, startTime, maxTime, isFinal);
        population.push_back(ind);}
    for (int i = 0; i < numRandom; ++i) {
        if (!isFinal && !checkTimeLimit(startTime, maxTime)) return population;
        Individual ind;
        ind.sequence = customers;
        shuffle(ind.sequence.begin(), ind.sequence.end(), rng);
        population.push_back(ind);}
    fitness.resize(populationSize);
    long long evalCount = 0;
    for (int i = 0; i < populationSize; ++i) {
        if (!isFinal && !checkTimeLimit(startTime, maxTime)) return population;
        fitness[i] = evaluateSolution(population[i], data, evalCount, numeric_limits<int>::max(), startTime, maxTime, isFinal);
        if (stopAlgorithm) return population;}
    //Find and print the best initial solution
    int bestIdx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    vector<Route> initialBestRoutes = reconstructRoutes(population[bestIdx].sequence, data, startTime, maxTime, true);
    printSolutionInfo(initialBestRoutes, data, "Initial Best Solution");
    return population;}

//Select parent
Individual selectParent(const vector<Individual>& population, const vector<double>& fitness, mt19937& rng, 
    chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return population[0];
    if (stopAlgorithm) return population[0];
    uniform_int_distribution<size_t> dist(0, population.size() - 1);
    size_t idx1 = dist(rng);
    Individual best = population[idx1];
    double bestFitness = fitness[idx1];
    for (int i = 1; i < 3; ++i) {
        size_t idx = dist(rng);
        Individual candidate = population[idx];
        double candidateFitness = fitness[idx];
        if (candidateFitness < bestFitness) {
            best = candidate;
            bestFitness = candidateFitness;}}
    return best;}

//Ordered Crossover (OX) operator
pair<Individual, Individual> crossover(const Individual& parent1, const Individual& parent2, mt19937& rng, 
    const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return {parent1, parent2};
    if (parent1.sequence.empty() || parent2.sequence.empty()) return {parent1, parent2};
    size_t n = parent1.sequence.size();
    Individual child1, child2;
    child1.sequence.resize(n);
    child2.sequence.resize(n);
    uniform_int_distribution<size_t> dist(0, n - 1);
    size_t start = dist(rng);
    size_t end = dist(rng);
    if (start > end) swap(start, end);
    for (size_t i = start; i <= end; ++i) {
        child1.sequence[i] = parent1.sequence[i];}
    size_t pos = (end + 1) % n;
    for (size_t i = 0; i < n; ++i) {
        int candidate = parent2.sequence[(end + 1 + i) % n];
        if (find(child1.sequence.begin() + start, child1.sequence.begin() + end + 1, candidate) == child1.sequence.begin() + end + 1) {
            child1.sequence[pos] = candidate;
            pos = (pos + 1) % n;
            if (pos == start) pos = end + 1;}}
    for (size_t i = start; i <= end; ++i) {
        child2.sequence[i] = parent2.sequence[i];}
    pos = (end + 1) % n;
    for (size_t i = 0; i < n; ++i) {
        int candidate = parent1.sequence[(end + 1 + i) % n];
        if (find(child2.sequence.begin() + start, child2.sequence.begin() + end + 1, candidate) == child2.sequence.begin() + end + 1) {
            child2.sequence[pos] = candidate;
            pos = (pos + 1) % n;
            if (pos == start) pos = end + 1;}}
    return {child1, child2};}

// Mutation operator
void mutate(Individual& ind, mt19937& rng, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime, bool isFinal = false) {
    if (!isFinal && !checkTimeLimit(startTime, maxTime)) return;
    if (stopAlgorithm) return;
    if (ind.sequence.size() < 2) return;
    uniform_int_distribution<size_t> dist(0, ind.sequence.size() - 1);
    size_t pos1 = dist(rng);
    size_t pos2 = dist(rng);
    if (pos1 > pos2) swap(pos1, pos2);
    reverse(ind.sequence.begin() + pos1, ind.sequence.begin() + pos2 + 1);}

//Dynamically set genetic algorithm parameters
void setGeneticAlgorithmParameters(const ProblemData& data, double& crossoverRate, double& mutationRate) {
    int n = data.customers.size() - 1;
    crossoverRate = min(0.85 + (n / 1000.0), 0.95);
    mutationRate = min(0.02 + (n / 5000.0), 0.1);}

// Genetic algorithm to optimize routes
vector<Route> geneticAlgorithm(const ProblemData& data, mt19937& rng,int populationSize, int maxGenerations,
                double crossoverRate, double mutationRate,int maxTime, int maxEvaluations,
                               chrono::steady_clock::time_point startTime) {
    stopAlgorithm = false;  // Reset stop flag
    timeLimitMessagePrinted = false;  // Reset time limit message flag
    long long evalCount = 0;  // Track number of evaluations
    // Create initial population and compute fitness
    vector<double> fitness;
    vector<Individual> population = generateInitialPopulation(data, rng,populationSize,startTime, maxTime, fitness);
    // Handle early termination or empty population
    if (stopAlgorithm || population.empty()) {
        return population.empty() ? vector<Route>() : reconstructRoutes(population[0].sequence, data, startTime, maxTime, true);}
    // Iterate through generations
    for (int generation = 0; generation < maxGenerations && !stopAlgorithm; ++generation) {
        if (!checkTimeLimit(startTime, maxTime)) break;  // Exit if time limit exceeded
        // Generate offspring population
        vector<Individual> offspring;
        vector<double> offspringFitness;
        while (offspring.size() < populationSize) {
            // Select parents via tournament selection
            Individual p1 = selectParent(population, fitness, rng, startTime, maxTime);
            Individual p2 = selectParent(population, fitness,rng,startTime, maxTime);
            // Apply crossover with given probability
            if (uniform_real_distribution<>(0,1)(rng) < crossoverRate) {
                auto [c1, c2] = crossover(p1, p2, rng, data, startTime, maxTime);  // Produce two children
                // Apply mutation with given probability
                if (uniform_real_distribution<>(0,1)(rng) < mutationRate) mutate(c1, rng, data, startTime, maxTime);
                if (uniform_real_distribution<>(0,1)(rng) < mutationRate) mutate(c2, rng, data, startTime, maxTime);
                // Add and evaluate first child
                offspring.push_back(c1);
                offspringFitness.push_back(evaluateSolution(c1, data, evalCount, maxEvaluations, startTime, maxTime));
                // Add and evaluate second child if space remains
                if (offspring.size() < populationSize) {
                    offspring.push_back(c2);
                    offspringFitness.push_back(evaluateSolution(c2, data, evalCount, maxEvaluations, startTime, maxTime));}
} else {
                // No crossover: add parents directly
                offspring.push_back(p1);
                offspringFitness.push_back(evaluateSolution(p1, data,evalCount,maxEvaluations,startTime, maxTime));
                if (offspring.size() < populationSize) {
                    offspring.push_back(p2);
                    offspringFitness.push_back(evaluateSolution(p2, data, evalCount, maxEvaluations, startTime, maxTime));
         }}   }

        // Combine parents and offspring for selection
        vector<pair<double, Individual>> combined;
        combined.reserve(2 * populationSize);
        for (int i = 0; i < populationSize; ++i)
            combined.emplace_back(fitness[i], population[i]);
        for (int i = 0; i < populationSize; ++i)
            combined.emplace_back(offspringFitness[i], offspring[i]);
        // Sort by fitness to select the best
        sort(combined.begin(), combined.end(), [](auto &a, auto &b){ return a.first < b.first; });
        // Update population with top individuals
        population.clear();
        fitness.clear();
        for (int i = 0; i < populationSize; ++i) {
            fitness.push_back(combined[i].first);
            population.push_back(combined[i].second);}
        // Track and display the best solution
        int bestIdx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
        vector<Route> currentBestRoutes = reconstructRoutes(population[bestIdx].sequence, data, startTime, maxTime, true);
        cout << "Generation " << generation + 1 << ":" << endl;
        printSolutionInfo(currentBestRoutes, data, "Best Solution");}
    // Return the best solution from final population
    int bestIdx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    return reconstructRoutes(population[bestIdx].sequence, data, startTime, maxTime, true);}

// Print solution
void printSolution(const vector<Route>& routes, const ProblemData& data, chrono::steady_clock::time_point startTime, int maxTime) {
    int routeNumber = 1;
    int vehicles = 0;
    double totalDistance = 0.0;
    cout << fixed << setprecision(2);
    for (size_t i = 0; i < routes.size(); ++i) {
        if (!routes[i].customerIds.empty()) {
            cout << "Route " << routeNumber++ << ":";
            for (int custId : routes[i].customerIds) {
                cout << " " << custId;}
            cout << "\n";
            vehicles++;
            totalDistance += routes[i].totalDistance;}}
    cout << "Vehicles: " << vehicles << "\n";
    cout << "Distance: " << totalDistance << "\n";}

//main function
int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0]
             << " <instanceFile> <maxTime> <maxEvaluations>"
             << endl;
        return 1;}
    const string instanceFile  = argv[1];
    const int  maxTime = stoi(argv[2]);
    const int  maxEvaluations = stoi(argv[3]);
    cout << "Running with instance: " << instanceFile
         << ", maxTime: "      << maxTime
         << ", maxEvaluations: "<< maxEvaluations << endl;
    ProblemData data = readInstance(instanceFile);
    const int populationSize = 75;
    const int maxGenerations = 100;
    double crossoverRate, mutationRate;
    setGeneticAlgorithmParameters(data, crossoverRate, mutationRate);
    cout << "Parameters: "
         << "populationSize = " << populationSize
         << ", maxGenerations = " << maxGenerations
         << ", crossoverRate = " << crossoverRate
         << ", mutationRate = " << mutationRate << endl;
    random_device rd;
    vector<uint32_t> seeds;
    for (int i = 0; i < 10; ++i) {
        seeds.push_back(rd());}
    auto now = chrono::high_resolution_clock::now();
    seeds.push_back(static_cast<uint32_t>(
        chrono::duration_cast<chrono::nanoseconds>(
            now.time_since_epoch()).count()));
    seed_seq seq(seeds.begin(), seeds.end());
    mt19937 rng(seq);
    auto startTime = chrono::steady_clock::now();
    vector<Route> solution = geneticAlgorithm(data,rng,populationSize,maxGenerations,crossoverRate,mutationRate,
        maxTime,maxEvaluations,startTime);
    auto endTime = chrono::steady_clock::now();
    double executionTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() / 1000.0;
    printSolution(solution, data, startTime, 0);
    if (solution.empty()) {   cout << "Final solution is empty." << endl;
    } else {   bool feasible = isSolutionFeasible(solution, data, startTime, 0, true);
        if (feasible) {cout << "Final solution is FEASIBLE." << endl;
        } else {cout << "Final solution is INFEasible!" << endl;}}
    cout << "Execution Time: " << fixed << setprecision(2) << executionTime << " seconds" << endl;
    return 0;}