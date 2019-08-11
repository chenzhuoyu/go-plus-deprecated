# Go Plus

**THIS IS A WORKING IN PROGRESS DRAFT SPECIFICATION**

Go Plus is designed to be a strict
super-set of Go programming language, which means:

* All valid Go code is valid Go Plus code

Go Plus works by translating Go Plus code into Go code, then compile
it with standard Go compiler.

## Ternary operator

```goplus
func Choose(c bool, v1, v2 int) int {
    return c ? v1 : v2
}
```

## Explicit access specifiers

```goplus
type SomeStruct struct {
    private ValueA      int // will be translated to `valueA`
    private valueB      int // will be kept as is
    public  someValue   int // will be translated to `SomeValue`
    public  PublicValue int // will be kept as is

    // public SomeValue int // illegal: duplicated with `someValue`
    // private valueA   int // illegal: duplicated with `ValueA`

    // Golang style access specifications are also supported
    OldStylePublicValue     int 
    oldStylePrivateValue    int
}

// the rules also apply to methods
public func (self SomeStruct) publicMethod() {}
private func (self SomeStruct) privateMethod() {}

// and types
private type InternalObject struct {
    // ...
}

// and global variables and consts
public var exportedValue int = 100
private const INTERNAL_CONST int = 10000
```

## Interface implementation validation

There is no way to enfore explicit interface implementation, cause
that breaks existing code.

The best thing we can do is to provide an optional mechanism to ensure
a certain struct implements certain interfaces.

```goplus
type ISomething interface {
    DoSomething()
}

// compiler ensures that `Something` must implement `ISomething`
type Something struct implements ISomething {
    // ...
}

func (Something) DoSomething() {
    println("hello, world")
}

// you can specify multiple interfaces at once
// the order is irrelevant
type OtherThing struct implements ISomething, error {
    // ...
}

func (OtherThing) DoSomething() {}
func (OtherThing) Error() string { return "" }

// illegal: `OtherThing` implements unspecified interface `io.Closer`
// func (OtherThing) Close() error { 
//     return nil
// }

// illegal: `CustomError` does not implement `error` interface
// type CustomError struct implements error {
//     // ...
// }
```

## Exception

```goplus
type SomeError struct {
    msg string
}

func (self SomeError) Error() string {
    return self.msg
}

func SomeFunction() {
    try {
        throw SomeError("whatever implements the error interface")
    } except (e error) {
        println("catch every error")
    } finally {
        println("do your cleanup")
    }
}
```

## C++ style template class

```goplus
template <T type>
type Stack struct {
    private sp    int
    private stack []T
}

public func (self Stack) push(val T) {
    // whatever pushes
}

// template placeholder can be primitive types
template <maxSize int, T type>
type LRUCache struct {
    private cache [maxSize]T
}
```

## C++ style template function

```goplus
template <T type>
func GetMax(v1, v2 T) T {
    return v1 > v2 ? v1 : v2
}
```

## C++ style polymorphism

It is very hard to implement multi-inheritance properly, so we decide
not to implement it. Single inheritance is good enough for most cases.

Java doesn't support multiple inheritance as well, yet it's still one 
of the world's most successful programming languages.

```goplus
type BaseAction class {
    // will generate a virtual table when a class
    // have at least 1 method that marked as `virtual`
}

// a non-virtual method, cannot be overridden
public func (self BaseAction) whatever() {
    // ...
}

// a virtual method that can be overridden
public virtual func (self BaseAction) init() {
    // ...
}

// a pure-virtual method without method body
public virtual func (self BaseAction) perform()

type SayHelloAction class extends BaseAction {
    // ...
}

// implements `BaseAction.perform`
public override func (self SayHelloAction) perform() {
    println("hello, world")
    self.BaseAction.init()       // call super class methods
    // self.BaseAction.perform() // illegal: pure-virtual method call 
}

// this throws a warning because it hides the method in super class
public func (self SayHelloAction) whatever() {
    // ...
}

// illegal: method `whatever` overrides nothing
// public override func (self SayHelloAction) whatever() {
//     // ...
// }
```

## Python-style function decorators

```goplus
func LoginRequired(handler func()) func() {
    return func() {
        // do login check, etc
        handler()
    }
}

@LoginRequired
public func RequestHandler() {
    // ...
}

// illegal: invalid function signature
// @LoginRequired
// public func InvalidFunc(a int) {
//     // ...
// }
```
