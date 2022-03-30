#pragma once

#include <vector>
using namespace std;

#include "vec3f.h"
#include "xyz-rgb.h"
#include <stdlib.h>

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))

class kDOP18 {
public:
	FORCEINLINE static void getDistances(const vec3f& p,
		double &d3, double &d4, double &d5, double &d6, double &d7, double &d8)
	{
		d3 = p[0] + p[1];
		d4 = p[0] + p[2];
		d5 = p[1] + p[2];
		d6 = p[0] - p[1];
		d7 = p[0] - p[2];
		d8 = p[1] - p[2];
	}

	FORCEINLINE static void getDistances(const vec3f& p, double d[])
	{
		d[0] = p[0] + p[1];
		d[1] = p[0] + p[2];
		d[2] = p[1] + p[2];
		d[3] = p[0] - p[1];
		d[4] = p[0] - p[2];
		d[5] = p[1] - p[2];
	}

	FORCEINLINE static double getDistances(const vec3f &p, int i)
	{
		if (i == 0) return p[0] + p[1];
		if (i == 1) return p[0] + p[2];
		if (i == 2) return p[1] + p[2];
		if (i == 3) return p[0] - p[1];
		if (i == 4) return p[0] - p[2];
		if (i == 5) return p[1] - p[2];
		return 0;
	}

public:
	double _dist[18];

	__device__ __host__ FORCEINLINE kDOP18() {
		empty();
	}

	__device__ __host__ FORCEINLINE kDOP18(const vec3f &v) {
		_dist[0] = _dist[9] = v[0];
		_dist[1] = _dist[10] = v[1];
		_dist[2] = _dist[11] = v[2];

		double d3, d4, d5, d6, d7, d8;
		getDistances(v, d3, d4, d5, d6, d7, d8);
		_dist[3] = _dist[12] = d3;
		_dist[4] = _dist[13] = d4;
		_dist[5] = _dist[14] = d5;
		_dist[6] = _dist[15] = d6;
		_dist[7] = _dist[16] = d7;
		_dist[8] = _dist[17] = d8;
	}

	__device__ __host__ FORCEINLINE kDOP18(const vec3f &a, const vec3f &b) {
		_dist[0] = MIN(a[0], b[0]);
		_dist[9] = MAX(a[0], b[0]);
		_dist[1] = MIN(a[1], b[1]);
		_dist[10] = MAX(a[1], b[1]);
		_dist[2] = MIN(a[2], b[2]);
		_dist[11] = MAX(a[2], b[2]);

		double ad3, ad4, ad5, ad6, ad7, ad8;
		getDistances(a, ad3, ad4, ad5, ad6, ad7, ad8);
		double bd3, bd4, bd5, bd6, bd7, bd8;
		getDistances(b, bd3, bd4, bd5, bd6, bd7, bd8);
		_dist[3] = MIN(ad3, bd3);
		_dist[12] = MAX(ad3, bd3);
		_dist[4] = MIN(ad4, bd4);
		_dist[13] = MAX(ad4, bd4);
		_dist[5] = MIN(ad5, bd5);
		_dist[14] = MAX(ad5, bd5);
		_dist[6] = MIN(ad6, bd6);
		_dist[15] = MAX(ad6, bd6);
		_dist[7] = MIN(ad7, bd7);
		_dist[16] = MAX(ad7, bd7);
		_dist[8] = MIN(ad8, bd8);
		_dist[17] = MAX(ad8, bd8);
	}

	__device__ __host__ FORCEINLINE bool overlaps(const kDOP18& b) const
	{
		for (int i = 0; i<9; i++) {
			if (_dist[i] > b._dist[i + 9]) return false;
			if (_dist[i + 9] < b._dist[i]) return false;
		}

		return true;
	}

	__device__ __host__ FORCEINLINE bool overlaps(const kDOP18 &b, kDOP18 &ret) const
	{
		if (!overlaps(b))
			return false;

		for (int i = 0; i<9; i++) {
			ret._dist[i] = MAX(_dist[i], b._dist[i]);
			ret._dist[i + 9] = MIN(_dist[i + 9], b._dist[i + 9]);
		}
		return true;
	}

	__device__ __host__ FORCEINLINE bool inside(const vec3f &p) const
	{
		for (int i = 0; i<3; i++) {
			if (p[i] < _dist[i] || p[i] > _dist[i + 9])
				return false;
		}

		double d[6];
		getDistances(p, d);
		for (int i = 3; i<9; i++) {
			if (d[i - 3] < _dist[i] || d[i - 3] > _dist[i + 9])
				return false;
		}

		return true;
	}

	__device__ __host__ FORCEINLINE kDOP18 &operator += (const vec3f &p)
	{
		_dist[0] = MIN(p[0], _dist[0]);
		_dist[9] = MAX(p[0], _dist[9]);
		_dist[1] = MIN(p[1], _dist[1]);
		_dist[10] = MAX(p[1], _dist[10]);
		_dist[2] = MIN(p[2], _dist[2]);
		_dist[11] = MAX(p[2], _dist[11]);

		double d3, d4, d5, d6, d7, d8;
		getDistances(p, d3, d4, d5, d6, d7, d8);
		_dist[3] = MIN(d3, _dist[3]);
		_dist[12] = MAX(d3, _dist[12]);
		_dist[4] = MIN(d4, _dist[4]);
		_dist[13] = MAX(d4, _dist[13]);
		_dist[5] = MIN(d5, _dist[5]);
		_dist[14] = MAX(d5, _dist[14]);
		_dist[6] = MIN(d6, _dist[6]);
		_dist[15] = MAX(d6, _dist[15]);
		_dist[7] = MIN(d7, _dist[7]);
		_dist[16] = MAX(d7, _dist[16]);
		_dist[8] = MIN(d8, _dist[8]);
		_dist[17] = MAX(d8, _dist[17]);

		return *this;
	}

	__device__ __host__ FORCEINLINE kDOP18 &operator += (const kDOP18 &b)
	{
		_dist[0] = MIN(b._dist[0], _dist[0]);
		_dist[9] = MAX(b._dist[9], _dist[9]);
		_dist[1] = MIN(b._dist[1], _dist[1]);
		_dist[10] = MAX(b._dist[10], _dist[10]);
		_dist[2] = MIN(b._dist[2], _dist[2]);
		_dist[11] = MAX(b._dist[11], _dist[11]);
		_dist[3] = MIN(b._dist[3], _dist[3]);
		_dist[12] = MAX(b._dist[12], _dist[12]);
		_dist[4] = MIN(b._dist[4], _dist[4]);
		_dist[13] = MAX(b._dist[13], _dist[13]);
		_dist[5] = MIN(b._dist[5], _dist[5]);
		_dist[14] = MAX(b._dist[14], _dist[14]);
		_dist[6] = MIN(b._dist[6], _dist[6]);
		_dist[15] = MAX(b._dist[15], _dist[15]);
		_dist[7] = MIN(b._dist[7], _dist[7]);
		_dist[16] = MAX(b._dist[16], _dist[16]);
		_dist[8] = MIN(b._dist[8], _dist[8]);
		_dist[17] = MAX(b._dist[17], _dist[17]);
		return *this;
	}

	__device__ __host__ FORCEINLINE kDOP18 operator + (const kDOP18 &v) const
	{
		kDOP18 rt(*this); return rt += v;
	}

	__device__ __host__ FORCEINLINE double length(int i) const {
		return _dist[i + 9] - _dist[i];
	}

	__device__ __host__ FORCEINLINE double width()  const { return _dist[9] - _dist[0]; }
	__device__ __host__ FORCEINLINE double height() const { return _dist[10] - _dist[1]; }
	__device__ __host__ FORCEINLINE double depth()  const { return _dist[11] - _dist[2]; }
	__device__ __host__ FORCEINLINE double volume() const { return width()*height()*depth(); }

	__device__ __host__ FORCEINLINE vec3f center() const {
		return vec3f(_dist[0] + _dist[9], _dist[1] + _dist[10], _dist[2] + _dist[11])*double(0.5);
	}

	__device__ __host__ FORCEINLINE double center(int i) const {
		return (_dist[i + 9] + _dist[i])*double(0.5);
	}

	__device__ __host__ FORCEINLINE void empty() {
		for (int i = 0; i<9; i++) {
			_dist[i] = FLT_MAX;
			_dist[i + 9] = -FLT_MAX;
		}
	}

	__device__ __host__ FORCEINLINE void dilate(double d) {
		//double sqrt2 = sqrt(2);
		double sqrt2 = 1.41421356237;
		for (int i = 0; i < 3; i++) {
			_dist[i] -= d;
			_dist[i + 9] += d;
		}
		for (int i = 0; i < 6; i++) {
			_dist[3 + i] -= sqrt2*d;
			_dist[3 + i + 9] += sqrt2*d;
		}
	}
};

#define kBOX kDOP18


class aap {
public:
	char _xyz;
	double _p;

	FORCEINLINE aap(const kBOX &total) {
		vec3f center = total.center();
		//如果深度最大
		char xyz = 2;

		//如果宽度最大
		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		}
		else//如果高度最大
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

		_xyz = xyz;
		//宽度或高度或深度的中心，以此作为划分依据？
		_p = center[xyz];
	}

	FORCEINLINE bool inside(const vec3f &mid) const {
		return mid[_xyz]>_p;
	}
};

class t_bvhTree;


class t_linearBvhNode {
public:
	kBOX _box;
	xyz2rgb _item;
	int _parent;
	int _left, _right;

	__device__ __host__ t_linearBvhNode() {
		_parent = _left = _right = -1;
	}

	__device__ __host__ FORCEINLINE bool isLeaf() { return _left == -1; }
	FORCEINLINE bool isRoot() { return _parent == -1; }

	__device__ __host__ void query(t_linearBvhNode *root, kBOX &bx, vector<xyz2rgb> &rets) {
		if (!_box.overlaps(bx))
			return;

		if (isLeaf()) {
			if (bx.inside(_item.xyz()))
				rets.push_back(_item);

			return;
		}

		root[_left].query(root, bx, rets);
		root[_right].query(root, bx, rets);
	}

	__device__ __host__ void query(t_linearBvhNode *root, kBOX &bx, xyz2rgb &dst, xyz2rgb &minPt, double &minDist) {
		if (!_box.overlaps(bx))
			return;

		if (isLeaf()) {
			if (bx.inside(_item.xyz())) {
				double dist = vdistance(_item.xyz(), dst.xyz());
				if (dist < minDist) {
					minDist = dist;
					minPt =_item;
				}
			}
			return;
		}

		root[_left].query(root, bx, dst, minPt, minDist);
		root[_right].query(root, bx, dst, minPt, minDist);
	}
};

class t_bvhNode {
public:
	kBOX _box;
	xyz2rgb _item;
	t_bvhNode *_parent;
	t_bvhNode *_left;
	t_bvhNode *_right;


	void Construct(t_bvhNode *p, xyz2rgb &pt)
	{
		_parent = p;
		_left = _right = NULL;
		_item = pt;
		_box += pt.xyz();
	}

public:
	t_bvhNode() {
		_parent = _left = _right = NULL;
	}

	t_bvhNode(t_bvhNode *p, xyz2rgb &pt) {
		Construct(p, pt);
	}

	t_bvhNode(t_bvhNode *p, vector<xyz2rgb>&pts)
	{
		if (pts.size() == 1) {
			Construct(p, pts[0]);
			return;
		}

		_parent = p;
		int num = pts.size();
		for (int i = 0; i < num; i++)
			_box += pts[i].xyz();

		if (num == 2) {
			_left = new t_bvhNode(this, pts[0]);
			_right = new t_bvhNode(this, pts[1]);
			return;
		}

		//重复寻找划分中心点
		aap pln(_box);
		vector<xyz2rgb> left, right;

		for (int i = 0; i < num; i++) {
			xyz2rgb &pt = pts[i];

			if (pln.inside(pt.xyz()))
				left.push_back(pt);
			else
				right.push_back(pt);
		}

		if (left.size() == 0)
		{
			left = std::vector<xyz2rgb>(
				std::make_move_iterator(right.begin() + right.size() / 2),
				std::make_move_iterator(right.end()));
			right.erase(right.begin() + right.size() / 2, right.end());
		}
		else if (right.size() == 0) {
			right = std::vector<xyz2rgb>(
				std::make_move_iterator(left.begin() + left.size() / 2),
				std::make_move_iterator(left.end()));
			left.erase(left.begin() + left.size() / 2, left.end());
		}

		_left = new t_bvhNode(this, left);
		_right = new t_bvhNode(this, right);
	}

	~t_bvhNode() {
		if (_left) delete _left;
		if (_right) delete _right;
	}

	FORCEINLINE t_bvhNode *getLeftChild() { return _left; }
	FORCEINLINE t_bvhNode *getRightChild() { return _right; }
	FORCEINLINE t_bvhNode *getParent() { return _parent; }

	FORCEINLINE bool getItem(xyz2rgb &item) {
		if (isLeaf()) {
			item = _item;
			return true;
		}
		else
			return false;
	}

	FORCEINLINE bool isLeaf() { return _left == NULL; }
	FORCEINLINE bool isRoot() { return _parent == NULL; }

	void query(kBOX &bx, vector<xyz2rgb> &rets) {
		if (!_box.overlaps(bx))
			return;

		if (isLeaf()) {
			if (bx.inside(_item.xyz()))
				rets.push_back(_item);

			return;
		}

		_left->query(bx, rets);
		_right->query(bx, rets);
	}

	//前序遍历，线性存储
	int store(t_linearBvhNode *data, int &idx, int pid) {
		int current = idx;

		data[current]._box = _box;
		data[current]._item = _item;
		data[current]._parent = pid;
		idx++;

		if (_left)
			data[current]._left = _left->store(data, idx, current);

		if (_right)
			data[current]._right = _right->store(data, idx, current);

		return current;
	}

	friend class t_bvhTree;
};

class t_linearBVH;

class t_bvhTree {
	t_bvhNode	*_root;
	int _numItems;
	int _maxPts;

	void Construct(vector<xyz2rgb> &data) {
		_numItems = data.size();

		kBOX total;
		for (int i = 0; i < data.size(); i++)
			total += data[i].xyz();

		aap pln(total);
		vector<xyz2rgb> left, right;
		//按是否大于中心划分点 划分bvh树
		for (int i = 0; i < data.size(); i++) {
			xyz2rgb &pt = data[i];

			if (pln.inside(pt.xyz()))
				left.push_back(pt);
			else
				right.push_back(pt);
		}

		//根节点的包围盒就是一整个
		_root = new t_bvhNode;
		_root->_box = total;

		if (data.size() == 1) {
			_root->_item = data[0];
			_root->_parent = NULL;
			_root->_left = _root->_right = NULL;
		}
		else {
			//如果没有左子树，划分一半右子树给左子树
			if (left.size() == 0)
			{
				left = std::vector<xyz2rgb>(
					std::make_move_iterator(right.begin() + right.size() / 2),
					std::make_move_iterator(right.end()));
				right.erase(right.begin() + right.size() / 2, right.end());
			}
			//反之亦然
			else if (right.size() == 0) {
				right = std::vector<xyz2rgb>(
					std::make_move_iterator(left.begin() + left.size() / 2),
					std::make_move_iterator(left.end()));
				left.erase(left.begin() + left.size() / 2, left.end());
			}

			_root->_left = new t_bvhNode(_root, left);
			_root->_right = new t_bvhNode(_root, right);
		}
	}

public:
	t_bvhTree(vector<xyz2rgb> &data) {
		_maxPts = 0;

		if (data.size())
			Construct(data);
		else
			_root = NULL;
	}

	~t_bvhTree() {
		if (!_root)
			return;

		delete _root;
	}

	int maxPts() const { return _maxPts; }

	bool query(xyz2rgb &pt, xyz2rgb &ret)
	{
		kBOX bound(pt.xyz());
		bound.dilate(1.5);

		vector<xyz2rgb> rets;
		query(bound, rets);
		//printf("find %d proximities\n", rets.size());
		if (rets.size() > _maxPts)
			_maxPts = rets.size();

		double minDist = 10000000;
		for (int i = 0; i < rets.size(); i++) {
			double dist = vdistance(rets[i].xyz(), pt.xyz());
			if (dist < minDist) {
				minDist = dist;
				ret = rets[i];
			}
		}

		return rets.size() > 0;
	}

	void query(kBOX &bx, vector<xyz2rgb> &rets)
	{
		_root->query(bx, rets);
	}

	kBOX box() {
		if (_root)
			return getRoot()->_box;
		else
			return kBOX();
	}

	FORCEINLINE t_bvhNode *getRoot() { return _root; }
	
	friend class t_bvhNode;
	friend class t_linearBVH;
};


class t_linearBVH {
public:
	t_linearBvhNode *_root;
	int _num;
	int _current;

public:
	int num() const { return _num; }
	void *data() { return _root; }

public:
	t_linearBVH(t_bvhTree *tree) {
		_num = tree->_numItems * 2 + 1;
		_root = new t_linearBvhNode[_num];
		_current = 0; //put from here

		tree->_root->store(_root, _current, -1);
	}

	void query(kBOX &bx, vector<xyz2rgb> &rets)
	{
		_root->query(_root, bx, rets);
	}

	__device__ __host__ void query(kBOX &bx, xyz2rgb &dst, xyz2rgb &minPt, float &minDist)
	{
		//_root->query(_root, bx, dst, minPt, minDist);
		
		// Allocate traversal stack from thread-local memory,
		//t_linearBvhNode stack[64];
		//t_linearBvhNode* stackPtr = stack;
		//t_linearBvhNode startNode;
		//startNode._parent = -2;
		//*stackPtr++ = startNode; // push
		int stack[64];
		int* stackPtr = stack;
		int startNode;
		startNode = -2;
		*stackPtr++ = startNode; // push

		// Traverse nodes starting from the root.
		t_linearBvhNode currentNode=*_root;
		int current_index = 0;
		if (currentNode.isLeaf())
		{
			if (bx.inside(currentNode._item.xyz())) {
				float dist = vdistance(currentNode._item.xyz(), dst.xyz());
				if (dist < minDist) {
					minDist = dist;
					minPt = currentNode._item;
				}
			}
			return;
		}
		while (current_index!=-2)
		{
			currentNode = _root[current_index];
			if (!currentNode._box.overlaps(bx))
			{
				current_index = *(--stackPtr);
				continue;
			}
			
			// Check each child node for overlap.
			t_linearBvhNode childL = _root[currentNode._left];
			t_linearBvhNode childR = _root[currentNode._right];
			bool overlapL = (childL._box.overlaps(bx));
			bool overlapR = (childR._box.overlaps(bx));

			// Query overlaps a leaf node => report collision.
			if (overlapL && childL.isLeaf())
			{
				if (bx.inside(childL._item.xyz())) {
					float dist = vdistance(childL._item.xyz(), dst.xyz());
					if (dist < minDist) {
						minDist = dist;
						minPt = childL._item;
					}
				}
			}

			if (overlapR && childR.isLeaf())
			{
				if (bx.inside(childR._item.xyz())) {
					float dist = vdistance(childR._item.xyz(), dst.xyz());
					if (dist < minDist) {
						minDist = dist;
						minPt = childR._item;
					}
				}
			}

			// Query overlaps an internal node => traverse.
			bool traverseL = (overlapL && !childL.isLeaf());
			bool traverseR = (overlapR && !childR.isLeaf());

			if (!traverseL && !traverseR)
				current_index = *(--stackPtr); // pop
			else
			{
				current_index = (traverseL) ? currentNode._left : currentNode._right;
				if (traverseL && traverseR)
				{
					*stackPtr++ = currentNode._right;
				}
			}
		}
		return;
	}

	__device__ __host__ bool query(xyz2rgb &pt, xyz2rgb &ret)
	{
		kBOX bound(pt.xyz());
		bound.dilate(0.1);
		//
		float minDist = 1000000;
		query(bound, pt, ret, minDist);
		//
		return minDist < 1000000;
	}
};
