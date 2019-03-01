"use strict";

let name;

/* Javascript notes: (https://javascript.info )
 * alert( `Hello, ${name}!` ); // Hello, John!
 * let value = true;
 * alert(typeof value); // boolean
 * value = String(value); // now value is a string "true"
 * alert(typeof value); // string
 * let str = "123";
 * alert(typeof str); // string
 * let num = Number(str); // becomes a number 123
 * alert(typeof num); // number
 * alert( Boolean("0") ); // true
 * alert( Boolean(" ") ); // spaces, also true (any non-empty string is true)
 * // No effect on numbers
 * let x = 1;
 * alert( +x ); // 1
 * let y = -2;
 * alert( +y ); // -2
 * // Converts non-numbers
 * alert( +true ); // 1
 * alert( +"" );   // 0
 * alert( 2 ** 3 ); // 8  (2 * 2 * 2)
 * alert( 0 === false ); // false, because the types are different
 * let test = prompt("Test", ''); // <-- for IE
 * result = confirm(question);
 * //Function confirm shows a modal window with a question and two buttons: OK and CANCEL.
 * let accessAllowed = (age > 18) ? true : false;  // accessAllowed is true if age>18, false otherwise
 * let accessAllowed = age > 18; // the same
 * outer: for (let i = 0; i < 3; i++) {
 *            for (let j = 0; j < 3; j++) {
 *                let input = prompt(`Value at coords (${i},${j})`, '');
 *                 // if an empty string or canceled, then break out of both loops
 *                 if (!input) break outer; // (*)
 *                 // do something with the value...
 *             }
 *         }
 * alert('Done!');
 * 
 * //Function definitions: In JavaScript, a function is a value and we can deal with that as a value. 
 * // The code above shows its string representation, that is the source code.
 * function sayHi() {
 *   // ...
 * }
 * 
 * let sayHi = function() {
 *   // ...
 * };
 * In other words, when JavaScript prepares to run the script or a code block, it first looks for 
 * Function Declarations in it and creates the functions. We can think of it as an “initialization stage”.
 * Function Expressions are created when the execution reaches them. So you can't call that is defined as
 * a function expression, before the assignment part is executed.
 * The Function Declaration is visible only inside the code block where it resides:
 * let age = prompt("What is your age?", 18);
 * if (age < 18) {
 *     function welcome() {
 *         alert("Hello!");
 *     }
 * 
 * } else {
 *     function welcome() {
 *         alert("Greetings!");
 *     }
 * }
 * welcome(); // Error: welcome is not defined
 * 
 * If above was created as a function expression, it would work
 * 
 * Arrow functions:
 * let func = (arg1, arg2, ...argN) => expression
 * // Same as:
 * let func = function(arg1, arg2, ...argN) {
 *    return expression;
 * }
 * 
 * Object declarations
 * let user = {
 *  name: "John",
 * age: 30,
 * "likes birds": true  // multiword property name must be quoted
 * };
 * 
 * let fruit = prompt("Which fruit to buy?", "apple");
 * let bag = {
 *    [fruit]: 5, // the name of the property is taken from the variable fruit
 * };
 * alert( bag.apple ); // 5 if fruit="apple"
 * 
 * function makeUser(name, age) {
 *    return {
 *      name, // same as name: name
 *      age   // same as age: age
 *    };
 *    
 * let user = { age: 30 };
 * let key = "age";
 * alert( key in user ); // true, takes the name from key and checks for such property
 * }
 * 
 * let user = {
 *    name: "John",
 *    age: 30,
 *    isAdmin: true
 * };
 * for(let key in user) {
 *    // keys
 *    alert( key );  // name, age, isAdmin
 *    // values for the keys
 *    alert( user[key] ); // John, 30, true
 * }
 * Are objects ordered? integer properties are sorted, others appear in creation order.
 * 
 * // copies all properties from permissions1 and permissions2 into user
 * Object.assign(user, permissions1, permissions2);
 *
 * let clone = Object.assign({}, user);
 * //for deep copy (otherwise clone = user would be a shallow copy, they would reference the same address)
 * use JavaScript library lodash, the method is called _.cloneDeep(obj). to clone objects that have objects in them
 * 
 * // method shorthand looks better, right?
 * let user = {
 *    sayHi() { // same as "sayHi: function()"
 *      alert("Hello");
 *    }
 * };
 * 
 *  //The value of this is the object “before dot”, the one used to call the method.
 * let user = {
 *    name: "John",
 *    age: 30,
 *    sayHi() {
 *       alert(this.name);
 *    }
 * };
 * user.sayHi(); // John
 * 
 * function handler() {
 * alert( 'Thanks!' );
 * }
 * input.addEventListener("click", handler);
 * // ....
 * input.removeEventListener("click", handler);
 * 
 * <button id="elem">Click me</button>
 * 
 *  //Event handler:
 * <script>
 *   class Menu {
 *     handleEvent(event) {
 *       // mousedown -> onMousedown
 *       let method = 'on' + event.type[0].toUpperCase() + event.type.slice(1);
 *       this[method](event);
 *     }
 * 
 *     onMousedown() {
 *       elem.innerHTML = "Mouse button pressed";
 *     }
 * 
 *     onMouseup() {
 *       elem.innerHTML += "...and released.";
 *     }
 *   }
 * 
 *   let menu = new Menu();
 *   elem.addEventListener('mousedown', menu);
 *   elem.addEventListener('mouseup', menu);
 * </script>
 * 
 */
