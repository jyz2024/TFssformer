#pragma once

#include <vector>
#include <utility>
#include <cstdint>
#include <cassert>
#include <thread>
#include <iostream>
#include <cmath>
#include <mutex>
#include "party3pc.h"
#include <omp.h>

using namespace std;

// Typedefs
typedef uint32_t myType;
typedef uint8_t smallType;
typedef std::pair<myType, myType> RSSMyType;
typedef std::pair<smallType, smallType> RSSSmallType;
typedef std::vector<RSSMyType> RSSVectorMyType;
typedef std::vector<RSSSmallType> RSSVectorSmallType;

// Constants
const int BIT_SIZE = 32;
const myType MINUS_ONE = (myType)-1;
const int PRIME_NUMBER = 67;
const int NO_CORES = 4;
const bool PARALLEL = false; // Set to false for simplicity in porting
const string SECURITY_TYPE = "Semi-honest"; // Default

// Globals
extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

// Helper to get partyNum
inline int getPartyNum() {
    return Party3PC::getInstance().party_id;
}
#define partyNum getPartyNum()

// Note about RSSVectorMyType a(size):
// `RSSVectorMyType` is a vector of `RSSMyType`, where `RSSMyType` is
// defined as `std::pair<myType, myType>`. Each pair represents the two
// local shares held by this party for a replicated-secret-shared value in
// the 3-party protocol. Writing `RSSVectorMyType a(size)` constructs a
// vector of `size` elements; each element is value-initialized to
// `{0,0}`. In `offline_llm3pc.cpp`, these pairs are then filled with the
// appropriate share values before calling `funcWrap`.

// Helper functions
inline size_t nextParty(size_t party) {
    return (party + 1) % 3;
}

inline size_t prevParty(size_t party) {
    return (party + 2) % 3;
}

inline void log_print(string str) {
    // cout << str << endl;
}

// Initialization
inline void initFalconCompat() {
    for (int i = 0; i < PRIME_NUMBER; i++) {
        for (int j = 0; j < PRIME_NUMBER; j++) {
            additionModPrime[i][j] = (i + j) % PRIME_NUMBER;
            subtractModPrime[i][j] = (i - j + PRIME_NUMBER) % PRIME_NUMBER;
            multiplicationModPrime[i][j] = (i * j) % PRIME_NUMBER;
        }
    }
}

// Math helpers
inline smallType wrapAround(myType a, myType b) {
    return (a > MINUS_ONE - b);
}

inline smallType wrap3(myType a, myType b, myType c) {
    myType temp = a + b;
    if (wrapAround(a, b))
        return 1 - wrapAround(temp, c);
    else
        return wrapAround(temp, c);
}

inline void wrap3(const RSSVectorMyType &a, const vector<myType> &b, vector<smallType> &c, size_t size) {
    for (size_t i = 0; i < size; ++i)
        c[i] = wrap3(a[i].first, a[i].second, b[i]);
}

template<typename T>
void addVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size) {
    for (size_t i = 0; i < size; ++i)
        c[i] = a[i] + b[i]; // Uses operator+ for pair
}

// Operator overloads for pair
template <typename T,typename U>                                                   
std::pair<T,U> operator+(const std::pair<T,U> & l,const std::pair<T,U> & r) {   
    return {l.first+r.first,l.second+r.second};
}

inline RSSSmallType addModPrime(RSSSmallType a, RSSSmallType b) {
    RSSSmallType ret;
    ret.first = additionModPrime[a.first][b.first];
    ret.second = additionModPrime[a.second][b.second]; 
    return ret;
}

inline smallType subConstModPrime(smallType a, smallType b) {
    return subtractModPrime[a][b];
}

inline RSSSmallType subConstModPrime(RSSSmallType a, const smallType r) {
    RSSSmallType ret = a;
    switch(partyNum) {
        case 0: ret.first = subtractModPrime[a.first][r]; break;       
        case 2: ret.second = subtractModPrime[a.second][r]; break;
    }		
    return ret;
}

inline RSSSmallType XORPublicModPrime(RSSSmallType a, bool r) {
    RSSSmallType ret;
    if (r == 0)
        ret = a;
    else {
        switch(partyNum) {
            case 0: 
                ret.first = subtractModPrime[1][a.first];
                ret.second = subtractModPrime[0][a.second];
                break;       
            case 1: 
                ret.first = subtractModPrime[0][a.first];
                ret.second = subtractModPrime[0][a.second];
                break;
            case 2: 
                ret.first = subtractModPrime[0][a.first];
                ret.second = subtractModPrime[1][a.second];
                break;
        }
    }
    return ret;
}

// Communication wrappers
template<typename T>
void sendVector(const vector<T> &vec, size_t target_id, size_t size) {
    // Forward to Party3PC::send_to which calls into MyNetwork::Peer. Using
    // this path ensures the underlying `SocketBuf`/`Peer` code performs the
    // actual send and updates bytes/round counters in real time.
    // `vec` is const here, but the network API expects a non-const pointer,
    // so use const_cast safely because we do not mutate the buffer.
    Party3PC::getInstance().send_to(target_id, const_cast<T*>(vec.data()), size, sizeof(T) * 8);
}

template<typename T>
void receiveVector(vector<T> &vec, size_t source_id, size_t size) {
    // Receive into the provided `vec`. The underlying Peer implementation
    // will update bytes/round counters so communication usage is tracked
    // live while the protocol runs.
    Party3PC::getInstance().recv_from(source_id, vec.data(), size, sizeof(T) * 8);
}

// Precompute class (Mock)
class Precompute {
public:
    void getShareConvertObjects(RSSVectorMyType &r, RSSVectorSmallType &shares_r, RSSVectorSmallType &alpha, size_t size) {
        // Mock implementation: Generate random shares
        for(size_t i=0; i<size; ++i) {
            myType val = rand();
            myType x1 = rand();
            myType x2 = rand();
            myType x3 = val - x1 - x2;
            
            if (partyNum == 0) r[i] = {x1, x2};
            else if (partyNum == 1) r[i] = {x2, x3};
            else if (partyNum == 2) r[i] = {x3, x1};
            
            // shares_r (bits of val)
            for(int j=0; j<BIT_SIZE; ++j) {
                smallType bit = (val >> (BIT_SIZE - 1 - j)) & 1;
                smallType s1 = rand() % PRIME_NUMBER;
                smallType s2 = rand() % PRIME_NUMBER;
                smallType s3 = (bit - s1 - s2 + 2 * PRIME_NUMBER) % PRIME_NUMBER;
                
                if (partyNum == 0) shares_r[i*BIT_SIZE + j] = {s1, s2};
                else if (partyNum == 1) shares_r[i*BIT_SIZE + j] = {s2, s3};
                else if (partyNum == 2) shares_r[i*BIT_SIZE + j] = {s3, s1};
            }
            
            // alpha (shares of 0)
            smallType a1 = rand() % 2;
            smallType a2 = rand() % 2;
            smallType a3 = 0 ^ a1 ^ a2;
            if (partyNum == 0) alpha[i] = {a1, a2};
            else if (partyNum == 1) alpha[i] = {a2, a3};
            else if (partyNum == 2) alpha[i] = {a3, a1};
        }
    }

    void getRandomBitShares(RSSVectorSmallType &eta, size_t size) {
        for(size_t i=0; i<size; ++i) {
            smallType val = rand() % 2;
            smallType s1 = rand() % 2;
            smallType s2 = rand() % 2;
            smallType s3 = val ^ s1 ^ s2;
            
            if (partyNum == 0) eta[i] = {s1, s2};
            else if (partyNum == 1) eta[i] = {s2, s3};
            else if (partyNum == 2) eta[i] = {s3, s1};
        }
    }
    
    void getSelectorBitShares(RSSVectorSmallType &c, RSSVectorMyType &m_c, size_t size) {
        // Not used in funcWrap but might be needed if I copy more
    }
};

extern Precompute PrecomputeObject;

// Forward declarations
void funcMultiplyNeighbours(const RSSVectorSmallType &c_1, RSSVectorSmallType &c_2, size_t size);
void funcCrunchMultiply(const RSSVectorSmallType &c, vector<smallType> &betaPrime, size_t size);
void funcCheckMaliciousDotProd(const RSSVectorSmallType &a, const RSSVectorSmallType &b, const RSSVectorSmallType &c, const vector<smallType> &temp, size_t size);

// Implementations

inline void funcCheckMaliciousDotProd(const RSSVectorSmallType &a, const RSSVectorSmallType &b, const RSSVectorSmallType &c, const vector<smallType> &temp, size_t size) {
    // No-op for semi-honest
}

inline void funcMultiplyNeighbours(const RSSVectorSmallType &c_1, RSSVectorSmallType &c_2, size_t size)
{
	assert (size % 2 == 0 && "Size should be 'half'able");
	vector<smallType> temp3(size/2, 0), recv(size/2, 0);
    #pragma omp parallel for
	for (int i = 0; i < size/2; ++i)
	{
		temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2*i].first][c_1[2*i+1].first]];
		temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2*i].first][c_1[2*i+1].second]];
		temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2*i].second][c_1[2*i+1].first]];
	}

	//Add random shares of 0 locally
	thread *threads = new thread[2];

	threads[0] = thread(sendVector<smallType>, ref(temp3), nextParty(partyNum), size/2);
	threads[1] = thread(receiveVector<smallType>, ref(recv), prevParty(partyNum), size/2);

	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;

    #pragma omp parallel for
	for (int i = 0; i < size/2; ++i)
	{
		c_2[i].first = temp3[i];
		c_2[i].second = recv[i];
	}
}

inline void funcCrunchMultiply(const RSSVectorSmallType &c, vector<smallType> &betaPrime, size_t size)
{
	size_t sizeLong = size*BIT_SIZE;
	RSSVectorSmallType c_0(sizeLong/2, make_pair(0,0)), c_1(sizeLong/4, make_pair(0,0)), 
					   c_2(sizeLong/8, make_pair(0,0)), c_3(sizeLong/16, make_pair(0,0)), 
					   c_4(sizeLong/32, make_pair(0,0)); 
	RSSVectorSmallType c_5(sizeLong/64, make_pair(0,0));

	vector<smallType> reconst(size, 0);

	funcMultiplyNeighbours(c, c_0, sizeLong);
	funcMultiplyNeighbours(c_0, c_1, sizeLong/2);
	funcMultiplyNeighbours(c_1, c_2, sizeLong/4);
	funcMultiplyNeighbours(c_2, c_3, sizeLong/8);
	funcMultiplyNeighbours(c_3, c_4, sizeLong/16);
	if (BIT_SIZE == 64)
		funcMultiplyNeighbours(c_4, c_5, sizeLong/32);

	vector<smallType> a_next(size), a_prev(size);
	if (BIT_SIZE == 64)
    #pragma omp parallel for
		for (int i = 0; i < size; ++i)
		{
			a_prev[i] = 0;
			a_next[i] = c_5[i].first;
			reconst[i] = c_5[i].first;
			reconst[i] = additionModPrime[reconst[i]][c_5[i].second];
		}
	else if (BIT_SIZE == 32)
    #pragma omp parallel for
		for (int i = 0; i < size; ++i)
		{
			a_prev[i] = 0;
			a_next[i] = c_4[i].first;
			reconst[i] = c_4[i].first;
			reconst[i] = additionModPrime[reconst[i]][c_4[i].second];
		}

	thread *threads = new thread[2];

	threads[0] = thread(sendVector<smallType>, ref(a_next), nextParty(partyNum), size);
	threads[1] = thread(receiveVector<smallType>, ref(a_prev), prevParty(partyNum), size);
	for (int i = 0; i < 2; i++)
		threads[i].join();
	delete[] threads;
    
    #pragma omp parallel for
	for (int i = 0; i < size; ++i)
		reconst[i] = additionModPrime[reconst[i]][a_prev[i]];

    #pragma omp parallel for
	for (int i = 0; i < size; ++i)
	{
		if (reconst[i] == 0)
			betaPrime[i] = 1;
	}
}

inline void funcDotProduct(const RSSVectorSmallType &a, const RSSVectorSmallType &b, 
							 RSSVectorSmallType &c, size_t size) 
{
    // Simplified implementation for funcPrivateCompare usage
    // Assuming semi-honest
    vector<smallType> temp3(size, 0), recv(size, 0);
    #pragma omp parallel for
    for(size_t i=0; i<size; ++i) {
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[a[i].first][b[i].first]];
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[a[i].first][b[i].second]];
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[a[i].second][b[i].first]];
    }
    
    thread *threads = new thread[2];
    threads[0] = thread(sendVector<smallType>, ref(temp3), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<smallType>, ref(recv), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++) threads[i].join();
    delete[] threads;
    
    #pragma omp parallel for
    for(size_t i=0; i<size; ++i) {
        c[i].first = temp3[i];
        c[i].second = recv[i];
    }
}

inline void funcPrivateCompare(const RSSVectorSmallType &share_m, const vector<myType> &r, 
							const RSSVectorSmallType &beta, vector<smallType> &betaPrime, 
							size_t size)
{
	log_print("funcPrivateCompare");
	assert(share_m.size() == size*BIT_SIZE && "Input error share_m");
	assert(r.size() == size && "Input error r");
	assert(beta.size() == size && "Input error beta");

	size_t sizeLong = size*BIT_SIZE;
	size_t index3, index2;
	RSSVectorSmallType c(sizeLong), diff(sizeLong), twoBetaMinusOne(sizeLong), xMinusR(sizeLong);
	RSSSmallType a, tempM, tempN;
	smallType bit_r;

    // Serial execution only for now
    #pragma omp parallel for
    for (int index2 = 0; index2 < size; ++index2)
    {
        //Computing 2Beta-1
        twoBetaMinusOne[index2*BIT_SIZE] = subConstModPrime(beta[index2], 1);
        twoBetaMinusOne[index2*BIT_SIZE] = addModPrime(twoBetaMinusOne[index2*BIT_SIZE], beta[index2]);

        for (size_t k = 0; k < BIT_SIZE; ++k)
        {
            index3 = index2*BIT_SIZE + k;
            twoBetaMinusOne[index3] = twoBetaMinusOne[index2*BIT_SIZE];

            bit_r = (smallType)((r[index2] >> (BIT_SIZE-1-k)) & 1);
            diff[index3] = share_m[index3];
                    
            if (bit_r == 1)
                diff[index3] = subConstModPrime(diff[index3], 1);
        }
    }

    //(-1)^beta * x[i] - r[i]
    funcDotProduct(diff, twoBetaMinusOne, xMinusR, sizeLong);

    #pragma omp parallel for
    for (int index2 = 0; index2 < size; ++index2)
    {
        a = make_pair(0, 0);
        for (size_t k = 0; k < BIT_SIZE; ++k)
        {
            index3 = index2*BIT_SIZE + k;
            c[index3] = a;
            tempM = share_m[index3];

            bit_r = (smallType)((r[index2] >> (BIT_SIZE-1-k)) & 1);

            tempN = XORPublicModPrime(tempM, bit_r);
            a = addModPrime(a, tempN);

            if (partyNum == 0)
            {
                c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
                c[index3].first = additionModPrime[c[index3].first][1];
                c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
            }
            else if (partyNum == 1)
            {
                c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
                c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
            }
            else if (partyNum == 2)
            {
                c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
                c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
                c[index3].second = additionModPrime[c[index3].second][1];
            }			
        }
    }

	RSSVectorSmallType temp_a(sizeLong/2), temp_b(sizeLong/2), temp_c(sizeLong/2);
	vector<smallType> temp_d(sizeLong/2);
	// if (SECURITY_TYPE.compare("Malicious") == 0)
	// 	funcCheckMaliciousDotProd(temp_a, temp_b, temp_c, temp_d, sizeLong/2);

	funcCrunchMultiply(c, betaPrime, size);	
}

inline void funcWrap(const RSSVectorMyType &a, RSSVectorSmallType &theta, size_t size)
{
    log_print("funcWrap");
    
    size_t sizeLong = size*BIT_SIZE;
    RSSVectorMyType x(size), r(size); 
    RSSVectorSmallType shares_r(sizeLong), alpha(size), beta(size), eta(size); 
    vector<smallType> delta(size), etaPrime(size); 
    vector<myType> reconst_x(size);

    PrecomputeObject.getShareConvertObjects(r, shares_r, alpha, size);
    addVectors<RSSMyType>(a, r, x, size);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        beta[i].first = wrapAround(a[i].first, r[i].first);
        x[i].first = a[i].first + r[i].first;
        beta[i].second = wrapAround(a[i].second, r[i].second);
        x[i].second = a[i].second + r[i].second;
    }

    vector<myType> x_next(size), x_prev(size);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        x_prev[i] = 0;
        x_next[i] = x[i].first;
        reconst_x[i] = x[i].first;
        reconst_x[i] = reconst_x[i] + x[i].second;
    }

    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(x_next), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<myType>, ref(x_prev), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        reconst_x[i] = reconst_x[i] + x_prev[i];

    wrap3(x, x_prev, delta, size); // All parties have delta
    PrecomputeObject.getRandomBitShares(eta, size);

    // cout << "PC: \t\t" << funcTime(funcPrivateCompare, shares_r, reconst_x, eta, etaPrime, size, BIT_SIZE) << endl;
    funcPrivateCompare(shares_r, reconst_x, eta, etaPrime, size);

    if (partyNum == 0)
    {   
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            theta[i].first = beta[i].first ^ delta[i] ^ alpha[i].first ^ eta[i].first ^ etaPrime[i];
            theta[i].second = beta[i].second ^ alpha[i].second ^ eta[i].second;
        }
    }
    else if (partyNum == 1)
    {   
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            theta[i].first = beta[i].first ^ delta[i] ^ alpha[i].first ^ eta[i].first;
            theta[i].second = beta[i].second ^ alpha[i].second ^ eta[i].second;
        }
    }
    else if (partyNum == 2)
    {   
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            theta[i].first = beta[i].first ^ alpha[i].first ^ eta[i].first;
            theta[i].second = beta[i].second ^ delta[i] ^ alpha[i].second ^ eta[i].second ^ etaPrime[i];
        }
    }
}
