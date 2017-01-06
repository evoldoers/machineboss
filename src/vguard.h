#ifndef VGUARD_INCLUDED
#define VGUARD_INCLUDED

#include <vector>
#include <iostream>
#include "stacktrace.h"

/* vector with accessor guards */
template<typename T>
class vguard : public std::vector<T> {
  typedef std::vector<T> _vgbase;
public:
  typedef typename _vgbase::value_type value_type;
  typedef typename _vgbase::size_type size_type;
  typedef typename _vgbase::allocator_type allocator_type;
  typedef typename _vgbase::reference reference;
  typedef typename _vgbase::const_reference const_reference;

  vguard (const allocator_type& a = allocator_type()) : _vgbase(a) { }
  vguard (size_t n, const value_type& t = value_type(), const allocator_type& a = allocator_type())
    : _vgbase(n,t,a) { }
  vguard (const _vgbase& v) : _vgbase(v) { }
  vguard (std::initializer_list<T> l) : _vgbase(l) { }
  template<typename InputIterator>
  vguard (InputIterator __first, InputIterator __last,
	  const allocator_type& a = allocator_type())
    : _vgbase(__first,__last,a) { }

  reference operator[] (size_type __n) {
#ifdef USE_VECTOR_GUARDS
    if (__n < 0 || __n >= this->size()) {
      std::cerr << "vector overflow: element " << __n << ", size is " << this->size() << std::endl;
      printStackTrace();
      throw;
    }
#endif  /* USE_VECTOR_GUARDS */
    return _vgbase::operator[] (__n);
  }

  const_reference operator[] (size_type __n) const {
#ifdef USE_VECTOR_GUARDS
    if (__n < 0 || __n >= this->size()) {
      std::cerr << "const vector overflow: element " << __n << ", size is " << this->size() << std::endl;
      printStackTrace();
      throw;
    }
#endif  /* USE_VECTOR_GUARDS */
    return _vgbase::operator[] (__n);
  }
};

#endif /* VGUARD_INCLUDED */
